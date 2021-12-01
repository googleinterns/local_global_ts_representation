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

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set()
import tensorflow as tf
from tensorflow_probability import distributions as tfd
# Import all kernel functions
from gl_rep.gp_kernel import *


class GLR(tf.keras.Model):
    def __init__(self, global_encoder, local_encoder, decoder, time_length, data_dim, window_size=20,
                 kernel='cauchy', beta=1., lamda=1., M=1, sigma=1.0, length_scale=1.0, kernel_scales=1, p=100):
        """
        Decoupled Global and Local Representation learning (GLR) model

        Attributes:
            global_encoder: Encoder model that learns the global representation for each time series sample
            local_encoder: Encoder model that learns the local representation of time series windows over time
            decoder: Decoder model that generated the time series sample distribution
            time_length: Maximum length of the time series samples
            data_dim: Input data dimension (number of features)
            window_size: Length of the time series window to learn representations for
            kernel: Gaussian Process kernels for different dimensions of local representations
            beta: KL divergence weight in loss term
            lamda: Counterfactual regularization weight in the loss term
            M: Number of Monte-Carlo samples
            lambda: Counterfactual regularization weight
            length_scale: Kernel length scale
            kernel_scales: number of different length scales over latent space dimensions
        """
        super(GLR, self).__init__()
        self.global_encoder = global_encoder
        self.local_encoder = local_encoder
        self.data_dim = data_dim
        self.decoder = decoder
        self.kernel = kernel
        self.time_length = time_length
        self.beta = beta
        self.lamda = lamda
        self.M = M
        self.p = p  # period for the periodic kernel
        self.kernel_scales = kernel_scales
        self.sigma = sigma
        self.window_size = window_size
        self.length_scale = length_scale
        self.prior = None
        self.latent_dim = local_encoder.zl_size
        self.pz_scale_inv = None
        self.pz_scale_log_abs_determinant = None

    def encode(self, x, m_mask=None):
        """Encode the local and global representations of a batch of time series sample
        Args:
            x: Batch of time series samples with shape [batch_size, T, feature_size]
            m_mask: Mask channel with the same size as x, indicating which samples are missing (1:missing 0: measured)
        """
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)
        if m_mask is not None:
            m_mask = tf.identity(m_mask)
            m_mask = tf.cast(m_mask, dtype=tf.float32)

        rnd_t = np.random.randint(0, int(x.shape[1] * 0.7))
        global_sample_len = int(x.shape[1] * 0.3)
        p_zg = self.global_encoder(x[:, rnd_t:rnd_t + global_sample_len, :],
                                  mask=None if m_mask is None else m_mask[:, rnd_t:rnd_t + global_sample_len, :])
        pz_t = self.local_encoder(x, mask=m_mask, window_size=self.window_size)
        z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
        return p_zg.mean(), z_t, pz_t

    def decode(self, z_t, z_g):
        """Generate the time series sample from the local and global representations
        Args:
            z_t: Local representations over time with shape [batch_size, time windows, local representation size]
            z_g: Global representation os time series sample [batch_size, global representation size]
        """
        x_hat_dist = self.decoder(z_t, z_g, output_len=self.window_size)
        return x_hat_dist

    def __call__(self, input):
        rnd_t = np.random.randint(0, input.shape[1] - 5 * self.window_size)
        z_g = self.global_encoder(input[:, rnd_t:rnd_t + 5 * self.window_size, :])
        pz_t = self.local_encoder(input, self.window_size)
        z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
        p_x_hat = self.decoder(z_t, z_g, output_len=self.window_size)
        return p_x_hat, z_t, z_g

    def _get_prior(self, time_length=None):
        """Estimate the prior over the local representations over time using the GP"""
        if time_length is None:
            time_length = self.time_length
        if self.kernel_scales>self.latent_dim:
            raise RuntimeError('Invalid kernel size')
        if self.prior is None:
            tiled_matrices = []
            kernel_dim = self.latent_dim // len(self.kernel)
            for i_k, kernel in enumerate(self.kernel):
                if i_k==len(self.kernel)-1:
                    kernel_dim = self.latent_dim - kernel_dim*(len(self.kernel)-1)
                # Compute kernel matrices for each latent dimension
                kernel_matrices = []
                for i in range(self.kernel_scales):
                    if kernel == "rbf":
                        kernel_matrices.append(rbf_kernel(time_length, self.length_scale / (2 ** i)))
                    elif kernel == "periodic":
                        kernel_matrices.append(periodic_kernel(time_length, self.length_scale / (2 ** i), self.p))
                    elif kernel == "diffusion":
                        kernel_matrices.append(diffusion_kernel(time_length, self.length_scale / (2 ** i)))
                    elif kernel == "matern":
                        kernel_matrices.append(matern_kernel(time_length, self.length_scale / (2 ** i)))
                    elif kernel == "cauchy":
                        kernel_matrices.append(cauchy_kernel(time_length, self.sigma, self.length_scale / (2 ** i)))
                # Combine kernel matrices for each latent dimension
                total = 0
                for i in range(self.kernel_scales):
                    if i == self.kernel_scales - 1:
                        multiplier = kernel_dim - total
                    else:
                        multiplier = int(kernel_dim // (self.kernel_scales))
                        total += multiplier
                    tiled_matrices.append(tf.tile(tf.expand_dims(kernel_matrices[i], 0), [multiplier, 1, 1]))
            kernel_matrix_tiled = tf.concat(tiled_matrices, axis=0)
            assert kernel_matrix_tiled.shape[0] == self.latent_dim
        white_noise = tf.eye(num_rows=time_length, num_columns=time_length, batch_shape=[self.latent_dim]) * 1e-5
        prior = tfd.MultivariateNormalFullCovariance(
            loc=tf.zeros([self.latent_dim, time_length], dtype=tf.float32),
            covariance_matrix=(kernel_matrix_tiled + white_noise))
        return prior

    def compute_loss(self, x, m_mask=None, x_len=None, return_parts=False, global_sample_len=None, is_train=True,):
        """Calculate the overall loss for a batch of samples x

                Loss = NLL + beta*(KL_divergence_local (GP prior) + KL_divergence_global) + lamda*counterfactual_regularization

                Args:
                    x: Batch of time series samples with shape [batch_size, T, feature_size]
                    m_mask: Mask channel with the same size as x, indicating which samples are missing (1:missing 0: measured)
                    x_len: Length of each time series sample
                    return_parts: Returns the overall loss if set to False, otherwise returns all the loss components
                    global_sample_len: Length of the time series sample to use for learning the global representation
                """
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.tile(x, [self.M, 1, 1])  # shape=(M*batch_size, time, dimensions)

        if global_sample_len is None:
            global_sample_len = int(x.shape[1] *0.3)
        if m_mask is not None:
            m_mask = tf.cast(m_mask, dtype=tf.float32)
            m_mask = tf.tile(m_mask, [self.M, 1, 1])  # shape=(M*batch_size, time, dimensions)
        if x_len is not None:
            x_len = tf.tile(x_len, [self.M])  # shape=(M*batch_size, time, dimensions)
            rnd_t = np.random.randint(0, max(1, (min(x_len) -global_sample_len)))
        else:
            rnd_t = np.random.randint(0, x.shape[1]-global_sample_len)

        pz = self._get_prior(time_length=x.shape[1] // self.window_size)
        p_zg = self.global_encoder(x[:, rnd_t:rnd_t + global_sample_len, :],
                                  mask=None if m_mask is None else m_mask[:, rnd_t:rnd_t + global_sample_len, :],
                                  is_train=is_train)
        z_g = p_zg.sample()
        pz_t = self.local_encoder(x, mask=m_mask, window_size=self.window_size)
        z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
        x_hat_dist = self.decoder(z_t, z_g, output_len=self.window_size)

        cf_loss = 0
        if self.lamda!=0:
            z_g_2 = tf.random.normal(shape=z_g.shape, stddev=1.)
            cf_dist = self.decoder(z_t, z_g_2, output_len=self.window_size)
            rnd_t_adv = np.random.randint(0, x.shape[1] - global_sample_len)
            pos_zg = self.global_encoder(cf_dist.sample()[:, rnd_t_adv:rnd_t_adv + global_sample_len, :], mask=None)
            cf_loss = tf.reduce_mean(tf.math.exp(pos_zg.log_prob(z_g)-pos_zg.log_prob(z_g_2)), -1)

        nll = -x_hat_dist.log_prob(x)  # shape=(M*batch_size, time, dimensions)
        kl = tfd.kl_divergence(pz_t, pz) / (x.shape[1] // self.window_size)
        kl = tf.reduce_mean(kl, 1)  # shape=(M*batch_size, time, dimensions)
        if m_mask is not None:
            nll = tf.where(m_mask == 1, tf.zeros_like(nll), nll)
            measured_ratio = (tf.reduce_sum(abs(tf.cast(m_mask, tf.float32) - 1), [1, 2]))/(m_mask.shape[1]*m_mask.shape[2])
            kl = kl * measured_ratio

        nll = tf.reduce_mean(nll, axis=[1,2])
        kl_zg = tf.reduce_sum(tfd.kl_divergence(p_zg, tfd.Normal(loc=0, scale=1.)),-1)
        elbo = -nll - self.beta * (kl + kl_zg) - self.lamda * cf_loss  # shape=(M*batch_size)
        elbo = tf.reduce_mean(elbo)

        if return_parts:
            nll = tf.reduce_mean(nll)
            kl = tf.reduce_mean(kl)
            cf_loss = tf.reduce_mean(cf_loss)
            kl_zg = tf.reduce_mean(kl_zg)
            return -elbo, nll, kl, cf_loss, kl_zg
        else:
            return -elbo

    def get_trainable_vars(self):
        """Get the trainable parameters of the graph"""
        self.compute_loss(x=tf.random.normal(shape=(10, self.time_length, self.data_dim), dtype=tf.float32),
                          m_mask=tf.zeros(shape=(10, self.time_length, self.data_dim), dtype=tf.float32))
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

    def _mi_upper_bound(self, x, m_mask=None):
        """Estimate the mutual information between the 2 set of representations"""
        assert len(x.shape) == 3, "Input should have shape: [batch_size, time_length, data_dim]"
        x = tf.identity(x)  # in case x is not a Tensor already...
        if m_mask is not None:
            m_mask = tf.identity(m_mask)  # in case m_mask is not a Tensor already...
            m_mask = tf.cast(m_mask, dtype=tf.float32)

        rnd_t = np.random.randint(0, int(x.shape[1] *0.7))
        global_sample_len = int(x.shape[1] *0.3)
        z_g = self.global_encoder(x[:, rnd_t:rnd_t + global_sample_len, :],
                                  mask=None if m_mask is None else m_mask[:, rnd_t:rnd_t + global_sample_len, :])
        pz_t = self.local_encoder(x, mask=m_mask, window_size=self.window_size)
        z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
        x_hat_dist = self.decoder(z_t, z_g, output_len=self.window_size)

        mi_upper_bound = tf.zeros((len(x),))
        mc_samples = 10
        rnd_inds = np.random.randint(0, len(x), size=(len(x), mc_samples))
        locs = pz_t.loc
        scale_trils = pz_t.scale_tril
        for ii in range(mc_samples):
            pz_hat = tfd.MultivariateNormalTriL(loc=tf.gather(locs, rnd_inds[:,ii]), scale_tril=tf.gather(scale_trils, rnd_inds[:,ii]))
            mi_upper_bound += tf.reduce_sum(tfd.kl_divergence(pz_t, pz_hat), axis=-1)
        mi_upper_bound = mi_upper_bound/mc_samples

        nll = -x_hat_dist.log_prob(x)
        if m_mask is not None:
            nll = tf.where(m_mask == 1, tf.zeros_like(nll), nll)
        nll = tf.reduce_sum(nll, [1,2])
        nll = tf.reduce_mean(nll)
        mi_upper_bound = tf.reduce_mean(mi_upper_bound, 0)
        mi = mi_upper_bound+nll
        return mi

    def train(self, trainset, validset, data, lr=1e-4, n_epochs=2):
        """Train the Global and Local representation learning (GLR) model
        Args:
            trainset: training dataset
            validset: validation dataset
            datae: Name of the dataset for training the model
            lr: learning rate
            n_epochs: Number of training epochs
        """
        _ = tf.compat.v1.train.get_or_create_global_step()
        trainable_vars = self.get_trainable_vars()
        optimizer = tf.keras.optimizers.Adam(lr)
        if not os.path.exists('./ckpt'):
            os.mkdir('./ckpt')
        summary_writer = tf.summary.create_file_writer("./logs/training_summary")
        with summary_writer.as_default():
            losses_train, losses_val = [], []
            kl_train, kl_val = [], []
            kl_zg_train, kl_zg_val = [], []
            nll_train, nll_val = [], []
            reg_train, reg_val = [], []
            for epoch in range(n_epochs + 1):
                epoch_loss, epoch_nll, epoch_kl, epoch_cf_reg, epoch_kl_zg = self.run_epoch(trainset, train=True,
                                                                                       optimizer=optimizer,
                                                                                       trainable_vars=trainable_vars)
                if epoch % 2 == 0:
                    print('=' * 30)
                    print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
                    losses_train.append(epoch_loss)
                    kl_train.append(epoch_kl)
                    kl_zg_train.append(epoch_kl_zg)
                    nll_train.append(epoch_nll)
                    reg_train.append(epoch_cf_reg)
                    print("Training loss = %.3f \t NLL = %.3f \t KL(local) = %.3f \t CF_reg = %.3f \t KL(zg) = %.3f"
                          % (epoch_loss, epoch_nll, epoch_kl, epoch_cf_reg, epoch_kl_zg))
                    epoch_loss, epoch_nll, epoch_kl, epoch_cf_reg, epoch_kl_zg = self.run_epoch(validset)
                    losses_val.append(epoch_loss)
                    kl_val.append(epoch_kl)
                    kl_zg_val.append(epoch_kl_zg)
                    nll_val.append(epoch_nll)
                    reg_val.append(epoch_cf_reg)
                    print("Validation loss = %.3f \t NLL = %.3f \t KL(local) = %.3f \t CF_reg = %.3f \t KL(zg) = %.3f"
                          % (epoch_loss, epoch_nll, epoch_kl, epoch_cf_reg, epoch_kl_zg))
                    self.save_weights('./ckpt/glr_%s_lambda%.1f' % (data, self.lamda))

            # Plot overall losses
            if not os.path.exists('./plots'):
                os.mkdir('./plots')
            plt.figure()
            plt.plot(losses_train, label='Train loss')
            plt.plot(losses_val, label='Validation loss')
            plt.legend()
            plt.savefig('./plots/glr_loss_%s_lambda%.1f.pdf' % (data, self.lamda))

            # Plot different components of the loss term
            f, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
            f.suptitle("Different segments of the loss term")
            for i, ax in enumerate(axs):
                if i == 0:
                    t_line = nll_train
                    v_line = nll_val
                    sub_title = "Negative Log Likelihood"
                if i == 1:
                    t_line = kl_train
                    v_line = kl_val
                    sub_title = "KL Divergence"
                if i == 2:
                    t_line = reg_train
                    v_line = reg_val
                    sub_title = "Counterfactual regularization"
                ax.plot(t_line, label='Train')
                ax.plot(v_line, label='Validation')
                ax.set_title(sub_title)
                ax.legend()
            f.tight_layout()
            plt.savefig('./plots/loss_components_%s_lambda%.1f.pdf' % (data, self.lamda))

    def run_epoch(self, dataset, optimizer=None, train=False, trainable_vars=None):
        """Training epoch for time series encoder and decoder models

        Args:
            dataset: Epoch dataset
            optimizer: tf Optimizer
            train: True if it is an epoch run for training, False is it is an inference run
            trainable_vars: List of trainable variables of the model
        """
        epoch_loss, epoch_kl, epoch_kl_zg, epoch_nll, epoch_cf_loss = [], [], [], [], []
        for i, batch in dataset.enumerate():
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            global_sample_len = int(0.4 * x_seq.shape[1])
            if train:
                with tf.GradientTape() as gen_tape:
                    gen_loss = self.compute_loss(x_seq, m_mask=mask_seq, x_len=x_lens,
                                                  global_sample_len=global_sample_len)
                gradients_of_generator = gen_tape.gradient(gen_loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients_of_generator, trainable_vars))
            loss, nll, kl, cf_loss, kl_zg = self.compute_loss(x_seq, m_mask=mask_seq, x_len=x_lens,
                                                               global_sample_len=global_sample_len,
                                                               return_parts=True)
            epoch_loss.append(loss.numpy())
            epoch_nll.append(nll.numpy())
            epoch_kl.append(kl.numpy())
            epoch_kl_zg.append(kl_zg)
            epoch_cf_loss.append(cf_loss)
        return np.mean(epoch_loss), np.mean(epoch_nll), np.mean(epoch_kl), np.mean(epoch_cf_loss), np.mean(epoch_kl_zg)
