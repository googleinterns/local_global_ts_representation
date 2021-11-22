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
import seaborn as sns
sns.set();
from sklearn.cluster import KMeans
import tensorflow as tf

from gl_rep.models import EncoderLocal, EncoderGlobal, WindowDecoder
from gl_rep.glr import GLR
from gl_rep.data_loaders import simulation_loader

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

np.random.seed(124)


def main(args):
    file_name = 'glr_%s_lambda%.1f' %(args.data, args.lamda)
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]
    if args.data=='simulation':
        trainset, validset, testset, normalization_specs, _ = simulation_loader(normalize='mean_zero', mask_threshold=0.00)
        zt_encoder = EncoderLocal(zl_size=configs["zl_size"], hidden_sizes=configs["glr_local_encoder_size"])
        zg_encoder = EncoderGlobal(zg_size=configs["zg_size"], hidden_sizes=configs["glr_global_encoder_size"])
        dec = WindowDecoder(output_size=configs["feature_size"],
                            output_length=configs["window_size"],
                            hidden_sizes=configs["glr_decoder_size"])
        rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, decoder=dec,
                        window_size=configs["window_size"], time_length=configs["t_len"],
                        data_dim=configs["feature_size"], kernel_scales=configs["kernel_scales"],
                        kernel=configs["kernels"], beta=0.5, M=configs["mc_samples"], sigma=.5,
                        lamda=FLAGS.lamda, length_scale=configs["length_scale"], p=15)
    rep_model.load_weights('./ckpt/%s' % file_name)

    for cv in range(1):
        all_zg = []
        all_zt = []
        all_labels, all_samples = [], []
        all_locals = []
        for i, batch in enumerate(testset):
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            z_g, z_t, pz_t = rep_model.encode(x_seq, mask_seq)
            all_zt.append(z_t)
            all_zg.append(z_g)
            all_labels.append(batch[3])
            all_locals.append(batch[4])
            all_samples.append(x_seq)
        all_zg = np.concatenate(all_zg, 0)
        all_zt = np.concatenate(all_zt, 0)
        all_labels = np.concatenate(all_labels, 0)
        all_locals = np.concatenate(all_locals, 0)
        all_samples = np.concatenate(all_samples, 0)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(all_zg)
        print('Cluster centers: ', kmeans.cluster_centers_)

        # Choose a random sample to plot (with zg=0 and zl=0)
        rnd_sample = np.argwhere(np.logical_and(all_labels==0, all_locals==0))[1][0]
        print('Sample ID: ', rnd_sample)
        # cf_global_class = kmeans.cluster_centers_[abs(1-global_class)]
        cf_global_class = all_zg[np.argwhere(np.logical_and(all_labels==1, all_locals==0))[1][0]]
        same_global_class = all_zg[rnd_sample]
        cf_local_class = all_zt[np.argwhere(np.logical_and(all_labels==1, all_locals==1))[1][0],:,:]
        cf_prediction_dist = rep_model.decoder(tf.expand_dims(all_zt[rnd_sample],0), tf.expand_dims(cf_global_class,0), output_len=window_size)
        reconstructed_prediction_dist = rep_model.decoder(tf.expand_dims(all_zt[rnd_sample],0), tf.expand_dims(all_zg[rnd_sample],0), output_len=window_size)

        global_cf_gt = gen_counterfactual(zg=abs(all_labels[rnd_sample]-1),zl=all_locals[rnd_sample])
        full_cf_gt = gen_counterfactual(zg=abs(all_labels[rnd_sample]-1),zl=abs(all_locals[rnd_sample]-1))
        local_cf_gt = gen_counterfactual(zg=all_labels[rnd_sample],zl=abs(all_locals[rnd_sample]-1))

        plt.figure(figsize=(15,5))
        plt.title('Counterfactual Simulation')
        plt.plot(all_samples[rnd_sample], label="Original signal")
        plt.plot(reconstructed_prediction_dist[0].mean(), label="Reconstructed signal")
        plt.plot(cf_prediction_dist[0].mean(), label="Generated counterfactual")
        plt.plot(global_cf_gt, label="Ground-truth counterfactual")
        plt.legend()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, ncol=4)
        plt.tight_layout()
        plt.savefig('./plots/counterfactual_sample_%s.pdf'%file_name)

        plt.figure()
        sns.histplot(all_zg.reshape(-1,), bins=100)
        plt.savefig('./plots/global_dist_sample_%s.pdf'%file_name)

        plt.figure()
        sns.histplot(all_zt.reshape(-1,), bins=100)
        plt.savefig('./plots/local_dist_sample_%s.pdf'%file_name)

        # Generate counterfactual samples by varying the local and global representation
        f, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,4))
        for i,p in enumerate([tf.expand_dims(same_global_class,0), [(same_global_class+cf_global_class)/2], tf.expand_dims(cf_global_class,0)]):
            for j,q in enumerate([tf.expand_dims(all_zt[rnd_sample],0), tf.expand_dims(cf_local_class,0)]):
                axs[0][j].set_title('Zl_%d'%j)
                axs[i][0].set_ylabel("Zg=%.2f"%(p[0][0]))
                axs[i][j].set_ylim(-3,+3)
                if i==0 and j==0:
                    axs[i][j].plot(all_samples[rnd_sample], label="Original signal", c='#8da0cb')
                    axs[i][j].legend()
                    axs[i][j].plot(rep_model.decoder(q, p, output_len=window_size).mean().numpy().reshape(-1,), c='#fc8d62')
                else:
                    axs[i][j].plot(rep_model.decoder(q, p, output_len=window_size).mean().numpy().reshape(-1,), c='#fc8d62')
                    if i==2 and j==1:
                        axs[i][j].plot(full_cf_gt, label="Ground-truth counterfactual", c='#66c2a5')
                        axs[i][j].legend()
                    if i==2 and j==0:
                        axs[i][j].plot(global_cf_gt,  c='#66c2a5')
                    if i==0 and j==1:
                        axs[i][j].plot(rep_model.decoder(q, p, output_len=window_size).mean().numpy().reshape(-1,),
                                       c='#fc8d62', label='Generated counterfactual')
                        axs[i][j].plot(local_cf_gt, c='#66c2a5')
                        axs[i][j].legend()
        plt.tight_layout()
        plt.savefig('./plots/cf_gradient_sample_%s.pdf'%file_name)


def gen_counterfactual(zg,zl, t_len=100, noise=0.1):
    """Generate a simulated sample based on the ground truth local and global representations

    Args:
        zg: Underlying global representations
        zl: Underlying local representations
        t_len: length of the time series sample
        noise: Varianc of Gaussian noise
    """
    t = np.array(np.arange(0, 10, 0.1))
    g_noise = np.random.randn(t_len) * noise
    if zg == 0:
        trend = t * 0.05
        c = np.ones((t_len,)) * -1.5
    elif zg == 1:
        trend = t * -0.05
        c = np.ones((t_len,)) * 1.5
    if zl == 0:
        seasonality = np.sin(40 * t / (2 * np.math.pi))*1.8
        ratio = 0.5
    elif zl == 1:
        seasonality = np.sin(20*t / (2 * np.math.pi))*1.2
        ratio = 0.5
    return ratio*(trend + c + seasonality) + g_noise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='simulation', help="dataset to use")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lamda', type=float, default=1., help="regularization weight")
    args = parser.parse_args()
    main(args)
