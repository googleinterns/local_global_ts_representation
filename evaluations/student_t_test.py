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
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set();

from gl_rep.models import *
from gl_rep.utils import *
from gl_rep.glr import GLR
from data.data_loaders import *

from scipy.stats import ttest_ind 


from absl import flags
from absl import app

FLAGS = flags.FLAGS

# Physionet config
# flags.DEFINE_integer('feature_size', 10, 'Input feature size')
# flags.DEFINE_list('local_encoder_sizes', [32, 64, 64], 'Layer sizes of the local encoder')
# flags.DEFINE_list('global_encoder_sizes', [32, 64, 64], 'Layer sizes of the global encoder')
# flags.DEFINE_list('e2e_encoder_sizes', [32, 64, 64], 'Layer sizes of the e2e encoder')
# flags.DEFINE_list('decoder_sizes', [256, 256], 'Layer sizes of the decoder')
# flags.DEFINE_integer('window_size', 4, 'Window size for the local encoder')
# flags.DEFINE_integer('t_len', 80, 'Maximum length of the time series sample')
# flags.DEFINE_integer('n_classes', 4, 'Number of classification classes')
# flags.DEFINE_string('data', 'physionet', 'Which dataset to use')
# flags.DEFINE_list('features', ['Coronary Care', 'Cardiac Surgery', 'Medical ICU', 'Surgical ICU'], 'Global feature list')

# HAR config
# flags.DEFINE_integer('feature_size', 561, 'Input feature size')
# flags.DEFINE_list('local_encoder_sizes', [128, 32], 'Layer sizes of the local encoder')
# flags.DEFINE_list('global_encoder_sizes', [128, 16], 'Layer sizes of the global encoder')
# flags.DEFINE_list('e2e_encoder_sizes', [128, 32, 16], 'Layer sizes of the e2e encoder')
# flags.DEFINE_list('decoder_sizes', [32, 128], 'Layer sizes of the decoder')
# flags.DEFINE_integer('window_size', 4, 'Window size for the local encoder')
# flags.DEFINE_integer('t_len', 200, 'Maximum length of the time series sample')
# flags.DEFINE_float('lr', 1e-3, 'Learning rate')
# flags.DEFINE_integer('repeat', 10, 'Number of random windows sampled from each data sample during the training')
# flags.DEFINE_integer('n_epochs', 100, 'Number of training epochs')
# flags.DEFINE_integer('n_classes', 6, 'Number of classification classes')
# flags.DEFINE_string('mode', 'glr', 'Which encoder baseline to use')
# flags.DEFINE_string('data', 'har', 'Which dataset to use')
# flags.DEFINE_string('category', 'local', 'Which representations to use for the classification')
# flags.DEFINE_string('experiment', 'classification', 'Which experiment setup to run')

# air quality config
flags.DEFINE_integer('feature_size', 10, 'Input feature size')
flags.DEFINE_list('local_encoder_sizes', [32, 64], 'Layer sizes of the local encoder')
flags.DEFINE_list('global_encoder_sizes', [32, 64], 'Layer sizes of the global encoder')
flags.DEFINE_list('e2e_encoder_sizes', [32, 32, 64], 'Layer sizes of the e2e encoder')
flags.DEFINE_list('decoder_sizes', [64, 32], 'Layer sizes of the decoder')
flags.DEFINE_integer('window_size', 24, 'Window size for the local encoder')
flags.DEFINE_integer('t_len', 24*28, 'Maximum length of the time series sample')
flags.DEFINE_integer('n_epochs', 120, 'Number of training epochs')
flags.DEFINE_integer('n_classes', 12, 'Number of classification classes')
flags.DEFINE_string('data', 'air_quality', 'Which dataset to use')
flags.DEFINE_list('features', ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 'Global feature list')


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


def main(_argv):
    for cv in range(1):
        if FLAGS.data=='air_quality':
            trainset, validset, testset, normalization_specs = airq_data_loader(normalize='mean_zero')
            file_name = 'glr_model_%s'%FLAGS.data
            zt_encoder = EncoderLocal(zl_size=8, hidden_sizes=[32, 64])  # , 128, 128, 64])
            zg_encoder = EncoderGlobal(zg_size=8, hidden_sizes=[32, 64])  # , 128, 64, 32])#, sample_len=24*7)
            dec = WindowDecoder(output_size=FLAGS.feature_size, output_length=FLAGS.window_size,
                                hidden_sizes=[64, 32])
            rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, discriminator=None, decoder=dec,
                            window_size=FLAGS.window_size, time_length=FLAGS.t_len, data_dim=FLAGS.feature_size, kernel_scales=4,
                            kernel=['cauchy', 'rbf'], beta=2., M=15, sigma=.5, lamda=5., length_scale=2., p=15)
            rep_model.load_weights('./ckpt/%s' % file_name)

        elif FLAGS.data=='physionet':
            trainset, validset, testset, normalization_specs = physionet_data_loader(normalize='mean_zero')
            file_name = 'glr_model_%s'%FLAGS.data
            zt_encoder = EncoderLocal(zl_size=8, hidden_sizes=FLAGS.local_encoder_sizes)
            zg_encoder = EncoderGlobal(zg_size=8, hidden_sizes=FLAGS.global_encoder_sizes)
            dec = WindowDecoder(output_size=FLAGS.feature_size, output_length=FLAGS.window_size, hidden_sizes=FLAGS.decoder_sizes)
            rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, discriminator=None, decoder=dec,
                            window_size=FLAGS.window_size, time_length=FLAGS.t_len, data_dim=FLAGS.feature_size, kernel_scales=4,
                            kernel=['cauchy', 'rbf'], beta=1., M=10, sigma=.5, lamda=2., length_scale=1., p=15)
            rep_model.load_weights('./ckpt/%s' % file_name).expect_partial()

        # p_zt_0, p_zt_1, p_zt_2, p_zt_3 = [], [], [], []
        p_zt = [[] for _ in range(FLAGS.n_classes)]
        for i, batch in enumerate(testset):
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            if FLAGS.data == 'physionet':
                labels = tf.reshape(batch[4][:, 3], shape=(-1,)) - 1
            elif FLAGS.data=='air_quality':
                labels = tf.convert_to_tensor([int(m.numpy().decode("utf-8"))-1. for m in batch[4][:, 1]], dtype=tf.float32)
            z_g, z_t, _ = rep_model.encode(x_seq, mask_seq)

            for classes in range(FLAGS.n_classes):
                inds = tf.where(labels==classes)
                sub_group = tf.gather_nd(z_t, inds).numpy()
                if len(sub_group)>0:
                    p_zt[classes].append(sub_group)

            # print(p_zt)
            # p_ind_0 = tf.where(labels==0)
            # p_ind_1 = tf.where(labels == 1)
            # p_ind_2 = tf.where(labels == 2)
            # p_ind_3 = tf.where(labels == 3)
            # p_zt_0.append(tf.gather_nd(z_t, p_ind_0).numpy())
            # p_zt_1.append(tf.gather_nd(z_t, p_ind_1).numpy())
            # p_zt_2.append(tf.gather_nd(z_t, p_ind_2).numpy())
            # p_zt_3.append(tf.gather_nd(z_t, p_ind_3).numpy())
        for classes in range(FLAGS.n_classes):
            p_zt[classes] = np.concatenate(p_zt[classes], 0)
        # p_zt_0 = np.concatenate(p_zt_0, 0)
        # p_zt_1 = np.concatenate(p_zt_1, 0)
        # p_zt_2 = np.concatenate(p_zt_2, 0)
        # p_zt_3 = np.concatenate(p_zt_3, 0)

        conf_matrix = np.zeros((FLAGS.n_classes, FLAGS.n_classes))
        for i, p in enumerate(p_zt):#[p_zt_0, p_zt_1, p_zt_2, p_zt_3]):
            for j, q in enumerate(p_zt):#[p_zt_0, p_zt_1, p_zt_2, p_zt_3]):
                pvals = []
                for dims in range(z_t.shape[-1]):
                    stats, pval = ttest_ind(p[:, :, dims], q[:, :, dims], nan_policy='raise')
                    pvals.extend(pval)
                conf_matrix[i,j] = np.mean(pvals)
        print('******* Pairwise p-values *******')
        print(conf_matrix)

        df_cm = pd.DataFrame(conf_matrix, index=[i for i in FLAGS.features],
                             columns=[i for i in FLAGS.features])
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True, vmin=0, vmax=1)
        plt.yticks(rotation=90)
        plt.xticks(rotation=0)
        plt.savefig("./plots/t_test_%s.pdf"%FLAGS.data)




if __name__=="__main__":
    app.run(main)