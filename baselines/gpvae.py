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
# Import all kernel functions
from gl_rep.gp_kernel import *

from gl_rep.models import EncoderLocal
from gl_rep.utils import train_vae, run_epoch
from gl_rep.glr import GLR
from gl_rep.data_loaders import airq_data_loader, simulation_loader, physionet_data_loader, har_data_loader



currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


class GPVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, time_length, data_dim, window_size=20,
                 kernel='cauchy', beta=1., M=1, sigma=1.0, length_scale=1.0, kernel_scales=1, p=100):
        """GPVAE model for learning representation for windows of time series

        based on the model introduced in
        Fortuin, V., Baranchuk, D., Rätsch, G. and Mandt, S., 2020, June.
        Gp-vae: Deep probabilistic time series imputation.
        In International conference on artificial intelligence and statistics (pp. 1651-1661). PMLR.

        Attributes:
            encoder: Encoder model for learning representation z_t for each window of time series
            decoder: Decoder model that generates a window of time series using the representation z_t
            time_length: Maximum possible length for the time series samples
            data_dim: Number of time series features
            window_size: Length of the time series window to encode
            kernel: Kernel of the GP prior
            beta: beta weight of the KL divergence based on beta-VAE models
            M: Number of Monte-Carlo samples
            length_scale: Length scale of the GP kernel
            kernel_scales: Number of different scales to use for the kernel priors over the different latent representations
        """
        super(GPVAE, self).__init__()
        self.local_encoder = encoder
        self.data_dim = data_dim
        self.decoder = decoder
        self.kernel = kernel
        self.time_length = time_length
        self.beta = beta
        self.M = M
        self.p = p  # period for the periodic kernel
        self.kernel_scales = kernel_scales
        self.sigma = sigma
        self.window_size = window_size
        self.length_scale = length_scale
        self.prior = None
        self.latent_dim = encoder.zl_size
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None

    def encode(self, x, m_mask=None):
        """Encode the time series sample into representations of each window over time"""
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        if m_mask is not None:
            m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = tf.cast(m_mask, dtype=tf.float32)
        pz_t = self.local_encoder(x, mask=m_mask, window_size=self.window_size)
        z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
        return z_t, pz_t

    def decode(self, z_t):
        """Generate the sample distribution using the local representations over time"""
        x_hat_dist = self.decoder(z_t)
        return x_hat_dist

    def __call__(self, input, m_mask=None):
        """Estimate the reconstructed time series sample distribution
        based on the conditional posterior of the representation"""
        if m_mask is not None:
            m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = tf.cast(m_mask, dtype=tf.float32)
        pz_t = self.local_encoder(input, mask=m_mask, window_size=self.window_size)
        z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
        p_x_hat = self.decoder(z_t)
        return p_x_hat, z_t

    def _get_prior(self, time_length=None):
        """Estimate the prior over the local representations over time using the GP"""
        if time_length is None:
            time_length = self.time_length
        if self.prior is None:
            tiled_matrices = []
            kernel_dim = self.latent_dim // len(self.kernel)
            for i_k, kernel in enumerate(self.kernel):
                if i_k == len(self.kernel) - 1:
                    kernel_dim = self.latent_dim - kernel_dim * (len(self.kernel) - 1)
                kernel_matrices = []
                for i in range(self.kernel_scales):
                    if kernel == "rbf":
                        kernel_matrices.append(rbf_kernel(time_length, self.length_scale / 2 ** i))
                    elif kernel == "periodic":
                        kernel_matrices.append(periodic_kernel(time_length, self.length_scale / 2 ** i, self.p))
                    elif kernel == "diffusion":
                        kernel_matrices.append(diffusion_kernel(time_length, self.length_scale / 2 ** i))
                    elif kernel == "matern":
                        kernel_matrices.append(matern_kernel(time_length, self.length_scale / 2 ** i))
                    elif kernel == "cauchy":
                        kernel_matrices.append(cauchy_kernel(time_length, self.sigma, self.length_scale / 2 ** i))
                # Combine kernel matrices for each latent dimension
                total = 0
                for i in range(self.kernel_scales):
                    if i == self.kernel_scales - 1:
                        multiplier = kernel_dim - total
                    else:
                        multiplier = int(np.ceil(kernel_dim / (self.kernel_scales)))
                        total += multiplier
                    tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
            kernel_matrix_tiled = tf.concat(tiled_matrices, axis=0)
            assert kernel_matrix_tiled.shape[0] == self.latent_dim
        white_noise = tf.eye(num_rows=time_length, num_columns=time_length, batch_shape=[self.latent_dim]) * 1e-5
        prior = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros([self.latent_dim, time_length], dtype=tf.float32),
            covariance_matrix=kernel_matrix_tiled + white_noise)
        return prior

    def compute_loss(self, x, m_mask=None, x_len=None, return_parts=False, **kwargs):
        """Calculate the overall GPVAE loss for a batch of samples x

        Loss = NLL + beta*KL_divergence (GP prior)

        Args:
            x: Batch of time series samples with shape [batch_size, T, feature_size]
            m_mask: Mask channel with the same size as x, indicating which samples are missing (1:missing 0: measured)
            x_len: Length of each time series sample
            return_parts: Returns the overall loss if set to False, otherwise returns all the loss components
        """
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        x = tf.tile(x, [self.M, 1, 1])  # shape=(M*BS, TL, D)

        if m_mask is not None:
            m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = tf.cast(m_mask, dtype=tf.float32)
            m_mask = tf.tile(m_mask, [self.M, 1, 1])  # shape=(M*BS, TL, D)
        if x_len is not None:
            x_len = tf.tile(x_len, [self.M])  # shape=(M*BS, TL, D)
        pz = self._get_prior(time_length=x.shape[1] // self.window_size)
        pz_t = self.local_encoder(x, mask=m_mask, window_size=self.window_size)
        z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
        x_hat_dist = self.decoder(z_t)

        nll = -x_hat_dist.log_prob(x)  # shape=(M*BS, TL, D)
        nll = tf.where(tf.math.is_finite(nll), nll, tf.ones_like(nll) * 500 )
        kl = tfd.kl_divergence(pz_t, pz) / (x.shape[1] // self.window_size)
        kl = tf.reduce_mean(kl, 1)  # shape=(M*BS)
        if m_mask is not None:
            nll = tf.where(m_mask == 1, tf.zeros_like(nll), nll)
            meassured_ratio = (tf.reduce_sum(abs(tf.cast(m_mask, tf.float32) - 1), [1, 2])) / (
                        m_mask.shape[1] * m_mask.shape[2])
            kl = kl * meassured_ratio

        nll = tf.reduce_mean(nll, axis=[1,2])
        elbo = -nll - self.beta * kl # shape=(M*BS)
        elbo = tf.reduce_mean(elbo)  # scalar1

        if return_parts:
            nll = tf.reduce_mean(nll)  # scalar
            kl = tf.reduce_mean(kl)  # scalar
            return -elbo, nll, kl, 0, 0
        else:
            return -elbo


    def get_trainable_vars(self):
        """Get the trainable parameters of the graph"""
        self.compute_loss(x=tf.random.normal(shape=(1, self.time_length, self.data_dim), dtype=tf.float32),
                          m_mask=tf.zeros(shape=(1, self.time_length, self.data_dim), dtype=tf.float32))
        return self.trainable_variables

    def get_conditional_predictive(self, z_l, prediction_steps=1):
        """Estimate the posterior distribution of the future local representations conditioned on the past observations"""
        history_used = z_l.shape[1]
        next_z = []
        prior = self._get_prior(time_length=history_used + prediction_steps)
        z_obs = z_l[:, -history_used:, :]
        mean = prior.mean()
        covariance = prior.covariance()

        mean_1 = mean[:, -prediction_steps:]
        mean_2 = mean[:, :-prediction_steps]
        cov_1_2 = covariance[:, -prediction_steps:, :-prediction_steps]
        cov_2_1 = covariance[:, :-prediction_steps, -prediction_steps:]
        cov_2_2 = covariance[:, :-prediction_steps, :-prediction_steps]
        cov_1_1 = covariance[:, -prediction_steps:, -prediction_steps:]

        for z_f in range(len(mean_1)):
            cov_mult = tf.matmul(cov_1_2[z_f], tf.linalg.inv(cov_2_2[z_f]))
            mean_cond = tf.expand_dims(tf.stack([mean_1[z_f]] * len(z_obs)), -1) + tf.matmul(
                tf.tile(tf.expand_dims(cov_mult, 0), [len(z_obs), 1, 1]),
                tf.expand_dims((z_obs[:, :, z_f] - mean_2[z_f]), -1))
            cov_cond = cov_1_1[z_f] - tf.matmul(cov_mult, cov_2_1[z_f])
            cond = tfd.MultivariateNormalTriL(loc=tf.squeeze(mean_cond, axis=-1), scale_tril=tf.linalg.cholesky(
                tf.tile(tf.expand_dims(cov_cond, 0), [len(z_obs), 1, 1])))
            z_pred = cond.sample()  # Shape=[batch, prediction_step]
            next_z.append(z_pred)
        next_z = tf.stack(next_z, axis=-1)  # Shape=[batch, prediction_step, zl_size]
        return next_z


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
    file_name = 'gpvae%d_%s' %(args.rep_size, args.data)

    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]
    if args.data == 'air_quality':
        n_epochs=100
        trainset, validset, testset, normalization_specs = airq_data_loader(normalize="mean_zero")
    elif args.data == 'physionet':
        n_epochs = 50
        trainset, validset, testset, normalization_specs = physionet_data_loader(normalize="mean_zero")
    elif args.data == 'har':
        n_epochs = 150
        trainset, validset, testset, normalization_specs = har_data_loader(normalize="mean_zero")

    # Create the representation learning models
    encoder = EncoderLocal(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
    decoder = Decoder(output_size=configs["feature_size"],
                      output_length=configs["window_size"],
                      hidden_sizes=configs["baseline_decoder_size"])
    rep_model = GPVAE(encoder, decoder, time_length=configs["t_len"], data_dim=configs["feature_size"],
                      window_size=configs["window_size"], kernel=['cauchy'], beta=1., M=1,
                      sigma=1.0, length_scale=2.0, kernel_scales=4, p=100)

    # Train the GPVAE baselines
    if args.train:
        print('Trainig GPVAE model on %s'%args.data)
        if is_continue:
            rep_model.load_weights('./ckpt/%s' %file_name)
        train_vae(rep_model, trainset, validset, lr=1e-3, n_epochs=n_epochs, data=args.data, file_name=file_name)
    rep_model.load_weights('./ckpt/%s' %file_name)
    test_loss, test_nll, test_kl, _, _ = run_epoch(rep_model, testset, args.data)
    print('\nGPVAE performance on %s data' % args.data)
    print('Loss = %.3f \t NLL = %.3f \t KL(local) = %.3f' % (test_loss, test_nll, test_kl))

    # Plot a reconstructed sample example from the test data
    for batch_i, batch in testset.enumerate():
        x_seq = batch[0]
        mask_seq, x_lens = batch[1], batch[2]

        p_xhat, _ = rep_model(x_seq, m_mask=mask_seq)
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
