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

""""Forecasting experiment: this experiment measures the forecasting
performance of the representation learning frameworks using the GP
prior over time"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set();
import tensorflow as tf

from gl_rep.models import EncoderLocal, EncoderGlobal, WindowDecoder
from gl_rep.glr import GLR
from gl_rep.data_loaders import airq_data_loader, simulation_loader, physionet_data_loader
from baselines.gpvae import GPVAE, Decoder


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

np.random.seed(124)
mode='glr'


def main(args):
    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]
    if args.data=='air_quality':
        trainset, validset, testset, normalization_specs = airq_data_loader(normalize='mean_zero')
    elif args.data == 'physionet':
        trainset, validset, testset, normalization_specs = physionet_data_loader(normalize='mean_zero')

    # Create the representation learning models
    if mode=='gpvae':
        file_name = 'gpvae%d_%s' %(args.rep_size, args.data)
        encoder = EncoderLocal(zl_size=args.rep_size, hidden_sizes=configs["baseline_encoder_size"])
        decoder = Decoder(output_size=configs["feature_size"],
                          output_length=configs["window_size"],
                          hidden_sizes=configs["baseline_decoder_size"])
        rep_model = GPVAE(encoder, decoder, time_length=configs["t_len"], data_dim=configs["feature_size"],
                          window_size=configs["window_size"], kernel=['cauchy'], beta=1., M=1,
                          sigma=1.0, length_scale=2.0, kernel_scales=4, p=100)
    elif mode=='glr':
        file_name = 'glr_%s_lambda%.1f' %(args.data, args.lamda)
        zt_encoder = EncoderLocal(zl_size=configs["zl_size"], hidden_sizes=configs["glr_local_encoder_size"])
        zg_encoder = EncoderGlobal(zg_size=configs["zg_size"], hidden_sizes=configs["glr_global_encoder_size"])
        dec = WindowDecoder(output_size=configs["feature_size"], output_length=configs["window_size"],
                            hidden_sizes=configs["glr_decoder_size"])
        rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, decoder=dec,
                        window_size=configs["window_size"], time_length=configs["t_len"],
                        data_dim=configs["feature_size"], kernel_scales=configs["kernel_scales"],
                        kernel=configs["kernels"], beta=configs["beta"], M=configs["mc_samples"], sigma=.5,
                        lamda=args.lamda, length_scale=configs["length_scale"], p=15)

    rep_model.load_weights('./ckpt/%s' % file_name)

    # Evaluate the forecasting performance on the test set
    forecast_window = 2 # Number of windows to predict
    overall_nll, overall_mse = [], []
    print('Running foresasting on %s'%file_name)
    for cv in range(args.cv):
        nll, mse = [], []
        for i, batch in enumerate(testset):
            x_seq = batch[0]
            mask_seq, x_lens = batch[1], batch[2]
            last_t = int(min(x_lens))
            start = int(last_t%configs["window_size"])
            x_future = x_seq[:, last_t-forecast_window*configs["window_size"]:last_t, :]
            prediction_map = mask_seq[:, last_t-forecast_window*configs["window_size"]:last_t, :]
            x_in = x_seq[:, start:last_t, :]
            mask_in = mask_seq[:, start:last_t, :]
            if mode == 'gpvae':
                z_t_past, pz_t_past = rep_model.encode(x_in[:, :-forecast_window * configs["window_size"], :],
                                                       mask_in[:, :-forecast_window * configs["window_size"], :])
                z_future = rep_model.get_conditional_predictive(z_t_past, prediction_steps=forecast_window)
                prediction_dist = rep_model.decoder(z_future)
                history_dist = rep_model.decoder(z_t_past)
            elif mode == 'glr':
                z_g, z_t_past, pz_t_past = rep_model.encode(x_in[:, :-forecast_window*configs["window_size"], :],
                                                            mask_in[:, :-forecast_window*configs["window_size"], :])
                z_future = rep_model.get_conditional_predictive(z_t_past, prediction_steps=forecast_window)
                prediction_dist = rep_model.decoder(z_future, z_g, output_len=configs["window_size"])
                history_dist = rep_model.decoder(z_t_past, z_g, output_len=configs["window_size"])
            nll.append(tf.reduce_sum(tf.where(prediction_map == 1, tf.zeros_like(prediction_dist.sample()), -prediction_dist.log_prob(x_future)).numpy())/tf.reduce_sum(abs(1-prediction_map)))
            mse.append(tf.reduce_sum(tf.where(prediction_map == 1, tf.zeros_like(prediction_dist.sample()), (prediction_dist.mean()-x_future)**2).numpy())/tf.reduce_sum(abs(1-prediction_map)))

        overall_nll.append(np.mean(nll))
        overall_mse.append(np.mean(mse))

    # Plot and example of a forecasted sample
    f, axs = plt.subplots(nrows=configs["feature_size"], ncols=1, figsize=(16, configs["feature_size"] * 2))
    t_axis = np.arange(x_in.shape[1])
    rnd_sample = np.argmin(tf.reduce_mean((prediction_dist.mean()-x_future)**2, [-1,-2])) # Pick the best sample fr plotting
    x_mean = prediction_dist.mean()
    x_std = prediction_dist.stddev()
    x_sample = prediction_dist.sample()
    for j, ax in enumerate(axs):
        ax.plot(t_axis[-forecast_window*configs["window_size"]:], x_mean[rnd_sample, :, j],  color='b', label='Forecasted signal')
        # ax.plot(t_axis, np.concatenate([history_dist.mean()[rnd_sample, :, j], x_mean[rnd_sample, :, j]]),
        #         '--', color='b', label='Reconstructed signal')
        ax.fill_between(t_axis[-forecast_window*configs["window_size"]:], (x_mean[rnd_sample, :, j] - x_std[rnd_sample, :, j]),
                        (x_mean[rnd_sample, :, j] + x_std[rnd_sample, :, j]), color='b', alpha=.2)
        ax.plot(t_axis, tf.where(mask_in[rnd_sample, :, j] == 1, np.nan, x_in[rnd_sample, :, j]), '*' , color='g', label='Original signal')
        ax.set_ylabel('%s' % (configs["feature_list"][j]))
        ax.axvline(x_in.shape[1]-forecast_window*configs["window_size"], color='r')
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("./plots/evaluations/forecasting_%s.pdf" % file_name)

    print('NLL: %.3f $\pm$ %.3f \t'%(np.mean(overall_nll), np.std(overall_nll)),
          'MSE:  %.3f $\pm$ %.3f'%(np.mean(np.mean(overall_mse)), np.std(overall_mse)))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='air_quality', help="dataset to use")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lamda', type=float, default=1., help="regularization weight")
    parser.add_argument('--cv', type=int, default=20, help="number of cross validation")
    parser.add_argument('--rep_size', type=int, default=8, help="Size of the representation vectors")
    args = parser.parse_args()
    if not os.path.exists('./plots/evaluations'):
        os.mkdir('./plots/evaluations')
    main(args)
