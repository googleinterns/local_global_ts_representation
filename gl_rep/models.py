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

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
# Import all kernel functions
from gl_rep.gp_kernel import *


class EncoderLocal(tf.keras.Model):
    """Encoder model to approximate the posterior distribution of the local representation of a time series sample

    The model is composed of a recurrent layer, followed by fully connected networks.
    It learns a vector of local representation Z_t for each window of time series

    Attributes:
        zl_size: Size of each local representation vector
        hidden_size: Hidden sizes of the model
    """
    def __init__(self, zl_size, hidden_sizes):
        """Initializes the instance"""
        super(EncoderLocal, self).__init__()
        self.zl_size = int(zl_size)
        self.hidden_sizes = hidden_sizes
        self.lstm = tf.keras.layers.LSTM(units=hidden_sizes[0], activation=tf.nn.sigmoid)
        if len(self.hidden_sizes) > 1:
            self.encoder_net = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
                                                    for h_i, h in enumerate(hidden_sizes[1:])])
        self.mean_estimator = tf.keras.layers.Dense(self.zl_size, dtype=tf.float32)
        self.covar_estimator = tf.keras.layers.Dense(2*self.zl_size, dtype=tf.float32)

    def __call__(self, x, mask=None, window_size=None):
        """Estimate the conditional posterior distribution of the local representation q(Z_l|X)"""
        zl_mean, zl_std = [], []
        if not mask is None:
            if x.shape[-1]>1:
                mask = (tf.reduce_sum(mask, axis=-1)<int(0.7*x.shape[-1]))
            else:
                mask = mask[:,:,0]==0
        for t in range(0, x.shape[1] - window_size + 1, window_size):
            if not mask is None:
                x_mapped = self.lstm(x[:, t:t + window_size, :], mask=mask[:, t:t + window_size])
            else:
                x_mapped = self.lstm(x[:, t:t + window_size, :])
            if len(self.hidden_sizes) > 1:
                x_mapped = self.encoder_net(x_mapped)
            zl_mean.append(self.mean_estimator(x_mapped))
            zl_std.append(tf.nn.softplus(self.covar_estimator(x_mapped)))
        zl_mean = tf.stack(zl_mean, axis=1)
        zl_mean = tf.transpose(zl_mean, perm=(0, 2, 1))
        zl_std = tf.stack(zl_std, axis=1)
        zl_covar = tf.transpose(zl_std, perm=(0, 2, 1))

        batch_size, _, time_length = zl_mean.shape
        covar_reshaped = tf.reshape(zl_covar, [batch_size, self.zl_size, 2 * time_length])

        dense_shape = [batch_size, self.zl_size, time_length, time_length]
        idxs_1 = np.repeat(np.arange(batch_size), self.zl_size * (2 * time_length - 1))
        idxs_2 = np.tile(np.repeat(np.arange(self.zl_size), (2 * time_length - 1)), batch_size)
        idxs_3 = np.tile(np.concatenate([np.arange(time_length), np.arange(time_length - 1)]), batch_size * self.zl_size)
        idxs_4 = np.tile(np.concatenate([np.arange(time_length), np.arange(1, time_length)]), batch_size * self.zl_size)
        idxs_all = np.stack([idxs_1, idxs_2, idxs_3, idxs_4], axis=1)

        with tf.device('/cpu:0'):
            # Obtain covariance matrix from precision one
            mapped_values = tf.reshape(covar_reshaped[:, :, :-1], [-1])
            prec_sparse = tf.sparse.SparseTensor(indices=idxs_all, values=mapped_values, dense_shape=dense_shape)
            prec_sparse = tf.sparse.reorder(prec_sparse)
            prec_tril = tf.compat.v1.sparse_add(tf.zeros(prec_sparse.dense_shape, dtype=tf.float32), prec_sparse)
            eye = tf.eye(num_rows=prec_tril.shape.as_list()[-1], batch_shape=prec_tril.shape.as_list()[:-2])
            prec_tril = prec_tril + eye
            cov_tril = tf.linalg.triangular_solve(matrix=prec_tril, rhs=eye, lower=False)
            cov_tril = tf.where(tf.math.is_finite(cov_tril), cov_tril, tf.zeros_like(cov_tril))

        num_dim = len(cov_tril.shape)
        perm = list(range(num_dim - 2)) + [num_dim - 1, num_dim - 2]
        cov_tril_lower = tf.transpose(cov_tril, perm=perm)
        return tfd.MultivariateNormalTriL(loc=zl_mean, scale_tril=cov_tril_lower)


class EncoderGlobal(tf.keras.Model):
    """Encoder model to approximate the conditional posterior of the local representation of a time series sample

    The model is composed of a recurrent layer, followed by fully connected networks.
    It learns a single vector for the global representation z_g of a time series sample

    Attributes:
        zg_size: Size of the global representation vector
        hidden_size: Hidden sizes of the model
    """
    def __init__(self, zg_size, hidden_sizes, sample_len=None):
        """Initializes the instance"""
        super(EncoderGlobal, self).__init__()
        self.zg_size = zg_size
        self.sample_len = sample_len
        self.hidden_sizes = hidden_sizes
        self.lstm = tf.keras.layers.LSTM(units=hidden_sizes[0])
        self.fc_net = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
                  for h in hidden_sizes[1:]])
        self.mean_gen = tf.keras.layers.Dense(self.zg_size, dtype=tf.float32)

    def __call__(self, x, mask=None, is_train=False):
        """Estimate the conditional posterior distribution of the global representation q(z_g|X)"""
        if not mask is None:
            if x.shape[-1]>1:
                mask = (tf.reduce_sum(mask, axis=-1)<int(0.7*x.shape[-1]))
            else:
                mask = mask[:,:,0]==0
            h_t = self.lstm(x , mask=mask)
        else:
            h_t = self.lstm(x)
        enc = self.fc_net(h_t)
        zg_mean = self.mean_gen(enc)
        return tfd.Normal(loc=zg_mean, scale=1.)


class WindowDecoder(tf.keras.Model):
    """Decoder model that approximates the likelihood distribution of the sample
    given the local and global representations q(X|z_g, Z_l)

    The model is composed of an embedding layer, a recurrent layer, followed by fully connected networks.

    Attributes:
        output_size: The number of features of the time series
        output_length: Length of the time series sample window
        hidden_size: Hidden sizes of the model
    """
    def __init__(self, output_size, output_length, hidden_sizes):
        """Initializes the instance"""
        super(WindowDecoder, self).__init__()
        self.output_size = output_size
        self.output_length = output_length
        self.hidden_sizes = hidden_sizes
        self.embedding = tf.keras.Sequential([tf.keras.layers.BatchNormalization(),
                                              tf.keras.layers.Dense(hidden_sizes[0], activation=tf.nn.tanh)])
        self.rnn = tf.keras.layers.LSTM(hidden_sizes[0], return_sequences=True)
        if len(hidden_sizes)>1:
            self.fc = tf.keras.Sequential([tf.keras.layers.Dense(h, activation=tf.nn.relu, dtype=tf.float32)
                                           for h in hidden_sizes[1:]])
        self.mean_gen = tf.keras.layers.Dense(self.output_size, dtype=tf.float32)
        self.cov_gen = tf.keras.layers.Dense(self.output_size, activation=tf.nn.sigmoid)

    def __call__(self, z_t, z_g, output_len):
        """Estimate the sample likelihood distribution conditioned on the local and global representations q(X|z_g, Z_l)"""
        n_batch, prior_len, _ = z_t.shape
        z = tf.concat([z_t, tf.tile(tf.expand_dims(z_g, axis=1), [1, z_t.shape[1], 1])], axis=-1)
        emb = self.embedding(z)
        recon_seq = []
        for t in range(z_t.shape[1]):
            rnn_out = self.rnn(inputs=tf.zeros((len(z_t), self.output_length, self.hidden_sizes[0])),
                               initial_state=[emb[:, t, :], emb[:, t, :]])
            recon_seq.append(rnn_out)
        recon_seq = tf.concat(recon_seq, 1)
        if len(self.hidden_sizes) > 1:
            recon_seq = self.fc(recon_seq)
        x_mean = self.mean_gen(recon_seq)
        x_cov = self.cov_gen(recon_seq)
        return tfd.Normal(loc=x_mean, scale=x_cov*0.5)

