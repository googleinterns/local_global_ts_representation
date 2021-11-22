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

""""Mortality prediction experiment with the Physionet ICU dataset
Using a simple recurrent model, we would like to predict in hospital
mortality from the time series representations learned over time."""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set()
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import sys
import tensorflow as tf

from gl_rep.models import EncoderLocal, EncoderGlobal, WindowDecoder
from gl_rep.glr import GLR
from gl_rep.data_loaders import physionet_data_loader
from baselines.gpvae import GPVAE, Decoder
from baselines.vae import VAE, Decoder, Encoder


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mode = 'glr'


def main(args):
    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)['physionet']
    n_epochs = 40
    trainset, validset, testset, normalization_specs = physionet_data_loader(normalize='mean_zero')

    # Create the representation learning models
    model = Predictor(32, [16])
    if mode=='supervised':
        file_name = './ckpt/e2e_mortality_predictor'
        rep_model_file = 'End to end'
        rep_model = None
    else:
        if mode=='glr':
            file_name = './ckpt/glr_mortality_predictor_lambda%.1f'%args.lamda
            rep_model_file = './ckpt/glr_physionet_lambda%.1f'%args.lamda
            zt_encoder = EncoderLocal(zl_size=configs["zl_size"], hidden_sizes=configs["glr_local_encoder_size"])
            zg_encoder = EncoderGlobal(zg_size=configs["zg_size"], hidden_sizes=configs["glr_global_encoder_size"])
            dec = WindowDecoder(output_size=configs["feature_size"], output_length=configs["window_size"],
                                hidden_sizes=configs["glr_decoder_size"])
            rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, decoder=dec,
                            window_size=configs["window_size"], time_length=configs["t_len"],
                            data_dim=configs["feature_size"], kernel_scales=configs["kernel_scales"],
                            kernel=configs["kernels"], beta=configs["beta"], M=configs["mc_samples"], sigma=.5,
                            lamda=args.lamda, length_scale=configs["length_scale"], p=15)
        elif mode=='gpvae':
            file_name = './ckpt/gpvae%d_mortality_predictor'%args.rep_size
            rep_model_file = './ckpt/gpvae%d_physionet'%args.rep_size
            encoder = EncoderLocal(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
            decoder = Decoder(output_size=configs["feature_size"],
                              output_length=configs["window_size"],
                              hidden_sizes=configs["baseline_decoder_size"])
            rep_model = GPVAE(encoder, decoder, time_length=configs["t_len"], data_dim=configs["feature_size"],
                              window_size=configs["window_size"], kernel=['cauchy'], beta=1., M=1,
                              sigma=1.0, length_scale=2.0, kernel_scales=4, p=100)
        elif mode=='vae':
            file_name = './ckpt/vae%d_mortality_predictor'%args.rep_size
            rep_model_file = './ckpt/vae%d_physionet'%args.rep_size
            encoder = Encoder(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
            decoder = Decoder(output_size=configs["feature_size"],
                              output_length=configs["window_size"],
                              hidden_sizes=configs["baseline_decoder_size"])
            rep_model = VAE(encoder=encoder, decoder=decoder, data_dim=configs["feature_size"], beta=1.,
                            M=configs["mc_samples"], sample_len=configs["t_len"])
        rep_model.load_weights(rep_model_file)

    if args.train:
        print(' =====> Trainig %s '%rep_model_file)
        if mode=='glr':
            lr = 1e-3
            model(tf.random.normal(shape=(5, 10, zt_encoder.zl_size), dtype=tf.float32),
                  tf.random.normal(shape=(5, zt_encoder.zl_size), dtype=tf.float32),
                  x_lens=tf.ones(shape=(5,))*10)
            optimizer = tf.keras.optimizers.Adam(lr)
        elif mode == 'supervised':
            lr = 1e-4
            model(tf.random.normal(shape=(5, 10, feature_size), dtype=tf.float32), None,
                  x_lens=tf.ones(shape=(5,))*10)
            optimizer = tf.keras.optimizers.Adam(lr)
        else:
            lr = 1e-3
            model(tf.random.normal(shape=(5, 10, encoder.zl_size), dtype=tf.float32), None,
                  x_lens=tf.ones(shape=(5,))*10)
            optimizer = tf.keras.optimizers.Adam(lr)
        trainable_vars = model.trainable_variables
        losses_train, acc_train, auroc_train = [], [], []
        losses_val, acc_val, auroc_val = [], [], []
        for epoch in range(n_epochs+1):
            epoch_loss, epoch_acc, epoch_auroc = run_epoch(model, trainset, rep_model, optimizer=optimizer,
                                                           train=True, trainable_vars=trainable_vars)
            if epoch % 5 == 0:
                print('=' * 30)
                print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
                losses_train.append(epoch_loss)
                acc_train.append(epoch_acc)
                auroc_train.append(epoch_auroc)
                print("Training loss = %.3f \t Accuracy = %.3f \t AUROC = %.3f" % (
                    epoch_loss, epoch_acc, epoch_auroc))
                epoch_loss, epoch_acc, epoch_auroc = run_epoch(model, validset, rep_model, train=False)
                losses_val.append(epoch_loss)
                acc_val.append(epoch_acc)
                auroc_val.append(epoch_auroc)
                print("Validation loss = %.3f \t Accuracy = %.3f \t AUROC = %.3f" % (
                    epoch_loss, epoch_acc, epoch_auroc))
        model.save_weights(file_name)
        plt.figure()
        plt.plot(losses_train, label='Train loss')
        plt.plot(losses_val, label='Validation loss')
        plt.legend()
        plt.savefig('./plots/evaluations/mortality_loss_%s.pdf' % (mode))

    if not mode=='supervised':
        rep_model.load_weights(rep_model_file)
    test_loss, test_acc, test_auroc = run_epoch(model, testset, rep_model, train=False)
    print("\n Test performance \t loss = %.3f \t AUPRC = %.3f \t AUROC = %.3f" % (
                        test_loss, test_acc, test_auroc))
    # Plot the risk estimation of a sample over time, along with the representations over time
    for batch_i, batch in testset.enumerate():
        x_seq = batch[0]
        mask_seq, x_lens = batch[1], batch[2]
        if mode=='glr':
            global_sample_len = int(x_seq.shape[1] * 0.3)
            rnd_t = np.random.randint(0, x_seq.shape[1] - global_sample_len)
            z_g = rep_model.global_encoder(x_seq[:, rnd_t:rnd_t + global_sample_len, :],
                                      mask=mask_seq[:, rnd_t:rnd_t + global_sample_len, :]).mean()
            pz_t = rep_model.local_encoder(x_seq, mask=mask_seq, window_size=rep_model.window_size)
            z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
        elif mode=='gpvae':
            z_t, _ = rep_model.encode(x_seq, mask_seq)
            z_g = None
        elif mode=='vae':
            z_t = rep_model.encoder(x_seq, mask_seq, rep_model.window_size).sample()
            z_g = None
        elif mode=='supervised':
            z_t = x_seq
            z_g = None
        predictions = []
        for t in range(1, z_t.shape[1]):
            predictions.append(model(z_t[:,:t], z_g, tf.ones_like(x_lens)*t)[:,0])
        predictions = tf.stack(predictions, axis=-1)
        rnd_sample = tf.math.argmax(tf.math.reduce_max(predictions, axis=-1) - tf.math.reduce_min(predictions, axis=-1))
        f, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 3))
        axs[0].plot(np.array(predictions[rnd_sample, :x_lens[rnd_sample]]))
        axs[0].set_ylim(0.0, 1.0)
        axs[0].set_title('Risk of mortality over time', fontweight='bold')
        axs[0].margins(x=0)
        sns.heatmap(np.array(z_t[rnd_sample, 1:x_lens[rnd_sample]]).T, cbar=False, linewidth=0.5, linewidths=0.05,
                    xticklabels=False, yticklabels=False, ax=axs[-1])
        axs[1].set_ylabel('Local Rep.')
        plt.tight_layout()
        plt.savefig("./plots/evaluations/mortality_prediction_over_time.pdf")
        break
    return test_loss, test_acc, test_auroc


class Predictor(tf.keras.Model):
    """Recurrent predictor model to estimate the risk of mortality in the ICU patients
    using the representations of time series over time

    Args:
        rnn_size: Hidden size of the RNN
        fc_sizes: List of the sizes of the fully connected layers
    """
    def __init__(self, rnn_size, fc_sizes):
        super(Predictor, self).__init__()
        self.rnn_size = rnn_size
        self.fc_sizes = fc_sizes

        self.rnn = tf.keras.layers.LSTM(rnn_size, return_sequences=True)
        self.fc = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
                                       for h in fc_sizes])
        self.prob = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, dtype=tf.float32)

    def __call__(self, local_encs, global_encs, x_lens):
        x_lens = tf.cast(x_lens, tf.int32)
        h = self.rnn(local_encs)
        h = tf.stack([h[i, x_lens[i]-1,:] for i in range(len(h))])
        if not global_encs is None:
            h = tf.concat([h, global_encs], axis=-1)
        logits = tf.keras.layers.Dropout(rate=0.5)(self.fc(h))
        probs = self.prob(logits)
        return probs

def run_epoch(model, dataset, glr_model, optimizer=None, train=False , trainable_vars=None):
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    epoch_loss, epoch_acc, epoch_auroc = [], [], []
    all_labels, all_predictions = [], []
    for batch_i, batch in dataset.enumerate():
        x_seq = batch[0]
        mask_seq, x_lens = batch[1], batch[2]
        labels = batch[4][:, -1]
        if mode=='glr':
            global_sample_len = int(x_seq.shape[1] * 0.3)
            rnd_t = np.random.randint(0, x_seq.shape[1] - global_sample_len)
            z_g = glr_model.global_encoder(x_seq[:, rnd_t:rnd_t + global_sample_len, :],
                                      mask=mask_seq[:, rnd_t:rnd_t + global_sample_len, :]).mean()
            pz_t = glr_model.local_encoder(x_seq, mask=mask_seq, window_size=glr_model.window_size)
            z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
            lens = x_lens//glr_model.window_size
        elif mode=='gpvae':
            z_t, _ = glr_model.encode(x_seq, mask_seq)
            z_g = None
            lens = x_lens//glr_model.window_size
        elif mode=='vae':
            z_t = glr_model.encoder(x_seq, mask_seq, glr_model.window_size).sample()
            z_g = None
            lens = x_lens//glr_model.window_size
        elif mode=='supervised':
            z_t = x_seq
            z_g = None
            lens = x_lens

        if train:
            with tf.GradientTape() as gen_tape:
                predictions = model(z_t, z_g, lens)
                loss = bce_loss(labels, predictions)
            grads = gen_tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
        else:
            predictions = model(z_t, z_g, lens)
            loss = bce_loss(labels, predictions)
        epoch_loss.append(loss.numpy())
        all_labels.append(labels)
        all_predictions.append(predictions)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    epoch_acc = average_precision_score(all_labels, all_predictions)
    epoch_auroc = (roc_auc_score(all_labels, all_predictions))
    return np.mean(epoch_loss), epoch_acc, epoch_auroc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lamda', type=float, default=1., help="regularization weight")
    parser.add_argument('--rep_size', type=int, default=8, help="Size of the representation vectors")
    args = parser.parse_args()
    if not os.path.exists('./plots/evaluations'):
        os.mkdir('./plots/evaluations')
    cv_loss, cv_acc, cv_auroc = [], [], []
    for cv in range(3):
        test_loss, test_acc, test_auroc = main(args)
        cv_loss.append(test_loss)
        cv_acc.append(test_acc)
        cv_auroc.append(test_auroc)
    print("***** Final performance *****")
    print("loss = %.3f $\pm$ %.3f \t AUPRC = %.3f $\pm$ %.3f \t AUROC = %.3f $\pm$ %.3f"%
          (np.mean(cv_loss), np.std(cv_loss), np.mean(cv_acc), np.std(cv_acc), np.mean(cv_auroc), np.std(cv_auroc)))
