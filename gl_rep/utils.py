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

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import random
import seaborn as sns
sns.set()
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import umap



def train_glr(model, trainset, validset, data, lr=1e-4, n_epochs=2):
    """Train the Global and Local representation learning (GLR) model

    Args:
        model: The glr model
        trainset: training dataset
        validset: validation dataset
        datae: Name of the dataset for training the model
        lr: learning rate
        n_epochs: Number of training epochs
    """
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()
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
        for epoch in range(n_epochs+1):
            epoch_loss, epoch_nll, epoch_kl, epoch_cf_reg, epoch_kl_zg = run_epoch(model, trainset, data, train=True,
                                                                                   optimizer=optimizer,
                                                                                   trainable_vars=trainable_vars)
            if epoch%2==0:
                print('='*30)
                print('Epoch %d'%epoch, '(Learning rate: %.5f)'%(lr))
                losses_train.append(epoch_loss)
                kl_train.append(epoch_kl)
                kl_zg_train.append(epoch_kl_zg)
                nll_train.append(epoch_nll)
                reg_train.append(epoch_cf_reg)
                print("Training loss = %.3f \t NLL = %.3f \t KL(local) = %.3f \t CF_reg = %.3f \t KL(zg) = %.3f"
                      %(epoch_loss, epoch_nll, epoch_kl, epoch_cf_reg, epoch_kl_zg))
                epoch_loss, epoch_nll, epoch_kl, epoch_cf_reg, epoch_kl_zg = run_epoch(model, validset, data)
                losses_val.append(epoch_loss)
                kl_val.append(epoch_kl)
                kl_zg_val.append(epoch_kl_zg)
                nll_val.append(epoch_nll)
                reg_val.append(epoch_cf_reg)
                print("Validation loss = %.3f \t NLL = %.3f \t KL(local) = %.3f \t CF_reg = %.3f \t KL(zg) = %.3f"
                      %(epoch_loss, epoch_nll, epoch_kl, epoch_cf_reg, epoch_kl_zg))
                model.save_weights('./ckpt/glr_%s_lambda%.1f' %(data, model.lamda))

        # Plot overall losses
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        plt.figure()
        plt.plot(losses_train, label='Train loss')
        plt.plot(losses_val, label='Validation loss')
        plt.legend()
        plt.savefig('./plots/glr_loss_%s_lambda%.1f.pdf' %(data, model.lamda))

        # Plot different components of the loss term
        f, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        f.suptitle("Different segments of the loss term")
        for i, ax in enumerate(axs):
            if i==0:
                t_line = nll_train
                v_line = nll_val
                sub_title = "Negative Log Likelihood"
            if i==1:
                t_line = kl_train
                v_line = kl_val
                sub_title = "KL Divergence"
            if i==2:
                t_line = reg_train
                v_line = reg_val
                sub_title = "Counterfactual regularization"
            ax.plot(t_line, label='Train')
            ax.plot(v_line, label='Validation')
            ax.set_title(sub_title)
            ax.legend()
        f.tight_layout()
        plt.savefig('./plots/loss_components_%s_lambda%.1f.pdf' %(data, model.lamda))


def run_epoch(model, dataset, data_type, optimizer=None, train=False , trainable_vars=None):
    """Training epoch for time series encoder and decoder models

    Args:
        model: The end to end encoder and decoder model
        dataset: Epoch dataset
        data_type: Name of the dataset for training the model
        optimizer: tf Optimizer
        train: True if it is an epoch run for training, False is it is an inference run
        trainable_vars: List of trainable variables of the model
    """
    epoch_loss, epoch_kl, epoch_kl_zg, epoch_nll, epoch_cf_loss = [], [], [], [], []
    for i, batch in dataset.enumerate():
        if data_type=='physionet':
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            global_sample_len = int(0.6*x_seq.shape[1]) # This is mainly because the missingness in Physionet data is a lot!
        elif data_type == 'air_quality' or data_type == 'har':
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            global_sample_len = int(0.3 * x_seq.shape[1])
        elif data_type == 'simulation':
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            global_sample_len = int(0.5 * x_seq.shape[1])
        elif data_type == 'har':
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            global_sample_len = int(0.5 * x_seq.shape[1])
        if train:
            with tf.GradientTape() as gen_tape:
                gen_loss = model.compute_loss(x_seq, m_mask=mask_seq, x_len=x_lens, global_sample_len=global_sample_len)
            gradients_of_generator = gen_tape.gradient(gen_loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients_of_generator, trainable_vars))
        loss, nll, kl, cf_loss, kl_zg = model.compute_loss(x_seq, m_mask=mask_seq, x_len=x_lens,
                                                                       global_sample_len=global_sample_len,
                                                                       return_parts=True)
        epoch_loss.append(loss.numpy())
        epoch_nll.append(nll.numpy())
        epoch_kl.append(kl.numpy())
        epoch_kl_zg.append(kl_zg)
        epoch_cf_loss.append(cf_loss)
    return np.mean(epoch_loss), np.mean(epoch_nll), np.mean(epoch_kl), np.mean(epoch_cf_loss), np.mean(epoch_kl_zg)


def plot_reps(dataset, gl_model, data):
    """Plot some visualizations for the representations

    This function generates 3 different types of plots:
    1- A scatter plot of the global representations of samples
    2- A heatmap of the local representation of a sample over time, alongside the GP prior
    3- An example of a reconstructed sample using the  GLR model

    Args:
        dataset: A dataset of tuples (x, mask, sample_len, local_representations, global_representations)
        gl_model: A trained GLR model
        data: Dataset type (air_quality, physionet, ...)
    """
    zg_all = []
    global_zs = []
    for i, batch in dataset.enumerate():
        if data=='air_quality':
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            local_z, global_z = batch[3], batch[4]
            global_features = ['year', 'month', 'station']
            local_features = ['Day', 'Rain']
            feature_list = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
        elif data=='simulation':
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            global_z = batch[3]
            global_features = ['zg']
        elif data == 'har':
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            local_z, global_z =  tf.expand_dims(batch[3], axis=-1), tf.expand_dims(batch[4], axis=-1)
            global_features = ['ID']
            local_features = ['State']
            feature_list = None
        elif data=='physionet':
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            local_z, global_z = batch[3], batch[4]
            global_features = ['Age', 'Gender', 'Height', 'ICUType', 'Weight', 'SAPS-I', 'SOFA', 'In-hospital_death']
            local_features = ['Mechanical Vent']
            feature_list = ['DiasABP', 'GCS', 'HCT', 'MAP', 'NIDiasABP', 'NIMAP', 'NISysABP', 'RespRate', 'SysABP', 'Temp']

        feature_size = x_seq.shape[-1]
        for _ in range(4):
            gwindow_len = int(x_seq.shape[1] *(0.6 if data=='physionet' else 0.3))
            rnd_t = np.random.randint(0, max(1, (min(x_lens) -gwindow_len)))
            z_g = gl_model.global_encoder(x_seq[:, rnd_t:rnd_t + gwindow_len, :], None if mask_seq is None else mask_seq[:, rnd_t:rnd_t + gwindow_len, :]).mean()
            zg_all.append(z_g)
            global_zs.append(global_z)
        if i==0:
            pz_t = gl_model.local_encoder(x_seq, mask=mask_seq, window_size=gl_model.window_size)
            pz = gl_model._get_prior(time_length=x_seq.shape[1] // gl_model.window_size)
            z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
            px_hat = gl_model.decoder(z_t, z_g, output_len=gl_model.window_size)
            x_mean = px_hat.mean()
            x_std = px_hat.stddev()
            rnd_ind = np.random.randint(len(x_seq))

            # Plot a reconstructed sample
            f, axs = plt.subplots(nrows=min(feature_size, 10), ncols=1, figsize=(11, min(feature_size, 10)*2))
            t_axis = np.arange(x_lens[rnd_ind])
            for i, ax in enumerate(axs):
                ax.plot(t_axis, x_mean[rnd_ind, :x_lens[rnd_ind], i], '--', label='Reconstructed signal')
                ax.fill_between(t_axis, (x_mean[rnd_ind, :x_lens[rnd_ind], i] - x_std[rnd_ind, :x_lens[rnd_ind], i]),
                                (x_mean[rnd_ind, :x_lens[rnd_ind], i] + x_std[rnd_ind, :x_lens[rnd_ind], i]),
                                color='b', alpha=.2)
                if mask_seq is None:
                    missing_x = x_seq[rnd_ind, :x_lens[rnd_ind], i]
                else:
                    missing_x = tf.where(mask_seq[rnd_ind, :x_lens[rnd_ind], i]==1, np.nan, x_seq[rnd_ind, :x_lens[rnd_ind], i])
                ax.plot(t_axis, missing_x, 'x', label='Original signal')
                if not feature_list is None:
                    ax.set_ylabel('%s' %(feature_list[i]))
                ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig("./plots/signal_reconstruction_%s_lambda%.1f.pdf" %(data, gl_model.lamda))

            # Plot local representations over time
            f, axs = plt.subplots(nrows=len(local_features)+2, ncols=1, figsize=(10, 2*(len(local_features)+2)))
            for ax_ind in range(len(local_features)):
                axs[ax_ind].plot(np.array(local_z[rnd_ind][:,ax_ind]))
                axs[ax_ind].set_title(local_features[ax_ind])
                axs[ax_ind].margins(x=0)
            sns.heatmap(np.array(z_t[rnd_ind]).T, cbar=True, linewidth=0.5,
                        linewidths=0.05, xticklabels=False, ax=axs[-1],
                        cbar_kws={'orientation': 'horizontal'})
            prior = tf.transpose(pz.sample(), perm=(1, 0))
            for prior_i in range(z_t.shape[-1]):
                axs[-2].plot(prior[:, prior_i])
            axs[-2].set_title("Prior")
            axs[-2].margins(x=0)
            plt.tight_layout()
            plt.savefig("./plots/local_representations_%s_lambda%.1f.pdf" %(data, gl_model.lamda))


    z_g = np.concatenate(zg_all)
    global_z = np.concatenate(global_zs)
    random_indices = np.random.choice(len(z_g), size=min(len(z_g),200), replace=False)
    z_g = np.array(z_g)[random_indices]
    global_z = global_z[random_indices]
    if data == 'air_quality':
        global_z = np.array(global_z)
        local_hash = {b'2': 'February', b'8': 'August', b'5': 'May', b'7': 'July', b'3': 'June', b'4': 'April',
                      b'10': 'October', b'6': 'June', b'12': 'December', b'11': 'November', b'9': 'September',
                      b'1': 'January'}
        global_z[:, 1] = list(map(lambda a: local_hash[a], global_z[:, 1]))
    global_embedded = umap.UMAP().fit_transform(z_g)
    # global_embedded = TSNE(n_components=2).fit_transform(np.array(z_g))
    # global_embedded = PCA(n_components=2).fit_transform(np.array(z_g))
    f = plt.figure(figsize=(6,6))
    if data=='air_quality':
        global_encoding = pd.DataFrame({"z1": global_embedded[:, 0],
                                    "z2": global_embedded[:, 1],
                                    global_features[2]: np.array(global_z[:, 2]),
                                    global_features[1]: global_z[:, 1]})
        g=sns.scatterplot(x="z1", y="z2", data=global_encoding, hue=global_features[1])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   fancybox=True, ncol=4)
    elif data=='har':
        global_encoding = pd.DataFrame({"z1": global_embedded[:, 0],
                                    "z2": global_embedded[:, 1],
                                    global_features[0]: global_z[:, 0]})
        g=sns.scatterplot(x="z1", y="z2", data=global_encoding, hue=global_features[0])#, ax=axs[0])
    elif data=='physionet':
        local_hash = ['Coronary Care Unit', 'Cardiac Surgery Recovery Unit', 'Medical ICU', 'Surgical ICU']
        global_encoding = pd.DataFrame({"z1": global_embedded[:, 0],
                                    "z2": global_embedded[:, 1],
                                    global_features[0]: global_z[:, 0],
                                    global_features[1]: global_z[:, 1],
                                    global_features[2]: global_z[:, 2],
                                    global_features[3]: np.array(list(map(lambda a: local_hash[int(a)-1], global_z[:, 3]))),
                                    global_features[-1]: global_z[:, -1]})
        g = sns.scatterplot(x="z1", y="z2", data=global_encoding, hue=global_features[3])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   fancybox=True, ncol=2)
    g.set(xlabel=None)
    plt.title('Global Representations', fontweight="bold")
    plt.tight_layout()
    plt.savefig("./plots/global_representations_%s_lambda%.1f.pdf" %(data, gl_model.lamda))


def train_vae(model, trainset, validset, data, file_name, lr=1e-4, n_epochs=2):
    """Function to train a VAE baselines

    Args:
        model: A VAE model composed of the encoder and decoder
        trainset: Training data
        validset: Validation data
        data: Dataset type (air_quality, physionet, ...)
        file_name: checkpoint name
        lr: Learning rate
        n_epochs: Number of training epochs
    """
    _ = tf.compat.v1.train.get_or_create_global_step()
    trainable_vars = model.get_trainable_vars()
    optimizer = tf.keras.optimizers.Adam(lr)
    if not os.path.exists('./ckpt'):
        os.mkdir('./ckpt')
    summary_writer = tf.summary.create_file_writer("./logs/vae_training_summary")
    with summary_writer.as_default():
        losses_train, losses_val = [], []
        kl_train, kl_val = [], []
        nll_train, nll_val = [], []
        for epoch in range(n_epochs+1):
            epoch_loss, epoch_nll, epoch_kl, _, _ = run_epoch(model, trainset, data, train=True,
                                                                    optimizer=optimizer,
                                                                    trainable_vars=trainable_vars)
            if epoch%1==0:
                print('='*30)
                print('Epoch %d'%epoch, '(Learning rate: %.5f)'%(lr))
                losses_train.append(epoch_loss)
                kl_train.append(epoch_kl)
                nll_train.append(epoch_nll)
                print("Training loss = %.3f \t NLL = %.3f \t KL = %.3f"%(epoch_loss, epoch_nll, epoch_kl))
                epoch_loss, epoch_nll, epoch_kl, _, _ = run_epoch(model, validset, data)
                losses_val.append(epoch_loss)
                kl_val.append(epoch_kl)
                nll_val.append(epoch_nll)
                print("Validation loss = %.3f \t NLL = %.3f \t KL(local) = %.3f"%(epoch_loss, epoch_nll, epoch_kl))
                model.save_weights('./ckpt/%s'%file_name)

        # Plot losses
        if not os.path.exists('./plots'):
            os.mkdir('./plots')
        plt.figure()
        plt.plot(losses_train, label='Train los')
        plt.plot(losses_val, label='Validation loss')
        plt.legend()
        plt.savefig('./plots/%s_loss.pdf'%(file_name))
        plt.figure()

        f, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        f.suptitle("Different segments of the loss term")
        for i, ax in enumerate(axs):
            if i==0:
                t_line = nll_train
                v_line = nll_val
                sub_title = "Negative Log Likelihood"
            if i==1:
                t_line = kl_train
                v_line = kl_val
                sub_title = "KL Divergence"
            ax.plot(t_line, label='Train')
            ax.plot(v_line, label='Validation')
            ax.set_title(sub_title)
            ax.legend()
        f.tight_layout()
        plt.savefig('./plots/%s_loss_components.pdf'%(file_name))