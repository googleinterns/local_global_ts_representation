# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
sns.set()
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from gl_rep.utils import train_vae, run_epoch
from gl_rep.glr import GLR
from gl_rep.data_loaders import airq_data_loader, simulation_loader, physionet_data_loader, har_data_loader

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


class VAE(tf.keras.Model):
    """Variational Auto Encoder for time series that learns a representation for each window of time series

    Attributes:
        encoder: Encoder model for learning representation z_t for each window of time series
        decoder: Decoder model that generates a window of time series using the representation z_t
        data_dim: Number of time series features
        beta: beta weight of the KL divergence based on beta-VAE models
        M: Number of Monte-Carlo samples
        sample_len: Maximum length of the time series samples
    """
    def __init__(self, encoder, decoder, data_dim, beta, M, sample_len):
        """Initializes the instance"""
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.M = M
        self.data_dim = data_dim
        self.latent_dim = encoder.zl_size
        self.sample_len = sample_len
        self.window_size = decoder.output_length

    def __call__(self, x_seq, mask_seq=None):
        """Estimate the reconstructed time series sample distribution
        based on the conditional posterior of the representation"""
        pz_t = self.encoder(x_seq, mask_seq, self.window_size)
        z_t = pz_t.sample()
        px_hat = self.decoder(z_t)
        return px_hat


    def compute_loss(self, x, m_mask=None, return_parts=False, **kwargs):
        """Calculate the overall VAE loss for a batch of samples x

        Loss = NLL + beta*KL_divergence

        Args:
            x: Batch of time series samples with shape [batch_size, T, feature_size]
            m_mask: Mask channel with the same size as x, indicating which samples are missing (1:missing 0: measured)
            x_len: Length of each time series sample
            return_parts: Returns the overall loss if set to False, otherwise returns all the loss components
        """
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)
        x = tf.tile(x, [self.M, 1, 1])  # shape=(M*BS, TL, D)

        if m_mask is not None:
            m_mask = tf.identity(m_mask)
            m_mask = tf.cast(m_mask, dtype=tf.float32)
            m_mask = tf.tile(m_mask, [self.M, 1, 1])  # shape=(M*BS, TL, D)

        pz_t = self.encoder(x, m_mask, self.window_size)
        z_t = pz_t.sample()
        px_hat = self.decoder(z_t)
        nll = -px_hat.log_prob(x)
        if m_mask is not None:
            nll = tf.where(m_mask == 1, tf.zeros_like(nll), nll)
        kl = tfd.kl_divergence(pz_t, tfd.MultivariateNormalDiag(loc=tf.zeros((z_t.shape[2])),
                                                                scale_diag=tf.ones((z_t.shape[2]))))

        kl = tf.reduce_mean(kl, axis=[-1])/(x.shape[2])
        nll = tf.reduce_mean(nll, axis=[1,2])

        elbo = -nll - self.beta * kl
        elbo = tf.reduce_mean(elbo)
        if return_parts:
            return -elbo, tf.reduce_mean(nll), tf.reduce_mean(kl), 0, 0
        return -elbo

    def get_trainable_vars(self):
        self.compute_loss(x=tf.random.normal(shape=(1, self.sample_len, self.data_dim), dtype=tf.float32),
                          m_mask=tf.zeros(shape=(1, self.sample_len, self.data_dim), dtype=tf.float32))
        return self.trainable_variables


class Encoder(tf.keras.Model):
    """Encoder model for time series that learns the representation of windows of samples

    The model is composed of a recurrent layer, followed by fully connected networks

    Attributes:
        zl_size: Size of the representation vector
        hidden_size: Hidden sizes of the model
    """
    def __init__(self, zl_size, hidden_sizes):
        """Initializes the instance"""
        super(Encoder, self).__init__()
        self.zl_size = int(zl_size)
        self.hidden_sizes = hidden_sizes

        self.lstm = tf.keras.layers.LSTM(units=hidden_sizes[0], activation=tf.nn.sigmoid)
        self.encoder_net = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
                                                for h_i, h in enumerate(hidden_sizes[1:])])
        self.mean_estimator = tf.keras.layers.Dense(self.zl_size, dtype=tf.float32)
        self.covar_estimator = tf.keras.layers.Dense(self.zl_size, dtype=tf.float32)

    def __call__(self, x, mask, window_size):
        """Encode a time series sample into a series of representations of each window

        Args:
            x: Batch of time series samples with shape [batch_size, T, feature_size]
            mask: Mask channel with the same size as x, indicating which samples are missing (1:missing 0: measured)
            window_size: Length of the window to learn representation for
        """
        zl_mean, zl_std = [], []
        if not mask is None:
            mask = (tf.reduce_sum(mask, axis=-1) < int(0.7 * x.shape[-1]))
        for t in range(0, x.shape[1] - window_size + 1, window_size):
            if not mask is None:
                x_mapped = self.lstm(x[:, t:t + window_size, :], mask=mask[:, t:t + window_size])
            else:
                x_mapped = self.lstm(x[:, t:t + window_size, :])
            x_mapped = self.encoder_net(x_mapped)
            zl_mean.append(self.mean_estimator(x_mapped))
            zl_std.append(tf.nn.softplus(self.covar_estimator(x_mapped)))
        zl_mean = tf.stack(zl_mean, axis=1)
        zl_cov = tf.stack(zl_std, axis=1)
        return tfd.MultivariateNormalDiag(loc=zl_mean, scale_diag=zl_cov)


class Decoder(tf.keras.Model):
    """Decoder model for time series that reconstructs windows of samples from their representations

        The model is composed of an embedding layer, a recurrent layer, followed by fully connected networks

        Attributes:
            output_size: Number of the time series features
            output_length: Length of each window to generate
            hidden_size: Hidden sizes of the model
        """
    def __init__(self, output_size, output_length, hidden_sizes, batch_norm=False):
        """Initializes the instance"""
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.output_length = output_length
        self.hidden_sizes = hidden_sizes
        self.batch_norm = batch_norm

        self.embedding = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.tanh, dtype=tf.float32)
                                              for h in hidden_sizes[0:1]])
        self.fc = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
                                       for h in hidden_sizes[1:]])
        self.rnn = tf.keras.layers.LSTM(hidden_sizes[0], return_sequences=True)
        self.mean_gen = tf.keras.layers.Dense(self.output_size, dtype=tf.float32)
        self.cov_gen = tf.keras.layers.Dense(self.output_size, activation=tf.nn.sigmoid)

    def __call__(self, z_t):
        """Reconstruct a time series from the representations of all windows over time

        Args:
            z_t: Representation of signal windows with shape [batch_size, n_windows, representation_size]
        """
        n_batch, prior_len, _ = z_t.shape
        emb = self.embedding(z_t)
        recon_seq = []
        for t in range(z_t.shape[1]):
            rnn_out = self.rnn(inputs=tf.random.normal(shape=(len(z_t), self.output_length, self.hidden_sizes[0])),
                               initial_state=[emb[:, t, :], emb[:, t, :]])
            recon_seq.append(rnn_out)
        recon_seq = tf.concat(recon_seq, 1)
        recon_seq = self.fc(recon_seq)
        x_mean = self.mean_gen(recon_seq)
        x_cov = self.cov_gen(recon_seq)
        return tfd.Normal(loc=x_mean, scale=x_cov*0.5)


def main(args):
    is_continue = False # Continue training an existing checkpoint
    file_name = 'vae%d_%s' %(args.rep_size, args.data)

    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]
    if args.data=='air_quality':
        n_epochs = 200
        trainset, validset, testset, normalization_specs = airq_data_loader(normalize="mean_zero")
    elif args.data=='physionet':
        n_epochs = 50
        trainset, validset, testset, normalization_specs = physionet_data_loader(normalize="mean_zero")
    elif args.data=='har':
        n_epochs = 150
        trainset, validset, testset, normalization_specs = har_data_loader(normalize='none')

    # Create the representation learning models
    encoder = Encoder(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
    decoder = Decoder(output_size=configs["feature_size"],
                      output_length=configs["window_size"],
                      hidden_sizes=configs["baseline_decoder_size"])
    rep_model = VAE(encoder=encoder, decoder=decoder, data_dim=configs["feature_size"], beta=1.,
                    M=configs["mc_samples"], sample_len=configs["t_len"])

    # Train the VAE baselines
    if args.train:
        print('Trainig VAE model on %s'%args.data)
        if is_continue:
            rep_model.load_weights('./ckpt/%s'%file_name)
        train_vae(rep_model, trainset, validset, lr=1e-3, n_epochs=n_epochs, data=args.data, file_name=file_name)

    # Report test performance
    rep_model.load_weights('./ckpt/%s'%file_name)
    test_loss, test_nll, test_kl, _, _ = run_epoch(rep_model, testset, args.data)
    print('\nVAE performance on %s data'%args.data)
    print('Loss = %.3f \t NLL = %.3f \t KL(local) = %.3f'%(test_loss, test_nll, test_kl))

    # Plot a reconstructed sample example from the test data
    for  batch_i, batch in testset.enumerate():
        x_seq = batch[0]
        mask_seq, x_lens = batch[1], batch[2]

        p_xhat = rep_model(x_seq, mask_seq=mask_seq)
        x_mean = p_xhat.mean()
        x_std = p_xhat.stddev()
        rnd_ind = np.random.randint(len(x_seq))

        f, axs = plt.subplots(nrows=min(10, configs["feature_size"]), ncols=1, figsize=(16, min(10, configs["feature_size"]) * 3))
        t_axis = np.arange(x_seq.shape[1])
        for i, ax in enumerate(axs):
            ax.plot(t_axis, x_mean[rnd_ind, :, i], '--', label='Reconstructed signal')
            ax.fill_between(t_axis, (x_mean[rnd_ind, :, i] - x_std[rnd_ind, :, i]),
                            (x_mean[rnd_ind, :, i] + x_std[rnd_ind, :, i]), color='b', alpha=.2)
            ax.plot(t_axis, x_seq[rnd_ind, :, i], label='Original signal')
            ax.set_ylabel('%s' % (configs["feature_list"][i]))
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("./plots/%s_signal_reconstruction.pdf" %(file_name))
        break
            

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='air_quality', help="dataset to use")
    parser.add_argument('--rep_size', type=int, default=8, help="Size of the representation vectors")
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    main(args)
