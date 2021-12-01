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
import os
import sys

from gl_rep.data_loaders import airq_data_loader, simulation_loader, physionet_data_loader, har_data_loader
from gl_rep.glr import GLR
from gl_rep.models import EncoderGlobal, EncoderLocal, WindowDecoder
from gl_rep.utils import plot_reps, train_glr
import tensorflow as tf

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)


def main(args):
    """
    Train and validate our local and global representation learning framework for different dataset
    """
    is_continue = False
    # Load the data and experiment configurations
    with open('configs.json') as config_file:
        configs = json.load(config_file)[args.data]
    if args.data=='air_quality':
        n_epochs = 250
        lr = 1e-3
        trainset, validset, testset, _ = airq_data_loader(normalize="mean_zero")
    elif args.data=='simulation':
        n_epochs = 100
        lr = 1e-2
        trainset, validset, testset, _, _ = simulation_loader(normalize="none", mask_threshold=0.0)
    elif args.data == 'physionet':
        n_epochs = 200
        lr = 1e-3
        trainset, validset, testset, _ = physionet_data_loader(normalize="mean_zero")
    elif args.data=='har':
        n_epochs = 150
        lr = 1e-3
        trainset, validset, testset, normalization_specs = har_data_loader(normalize='none')

    # Create the representation learning models
    zt_encoder = EncoderLocal(zl_size=configs["zl_size"], hidden_sizes=configs["glr_local_encoder_size"])
    zg_encoder = EncoderGlobal(zg_size=configs["zg_size"], hidden_sizes=configs["glr_global_encoder_size"])
    dec = WindowDecoder(output_size=configs["feature_size"], output_length=configs["window_size"],
                        hidden_sizes=configs["glr_decoder_size"])
    rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, decoder=dec,
                    window_size=configs["window_size"], time_length=configs["t_len"],
                    data_dim=configs["feature_size"], kernel_scales=configs["kernel_scales"],
                    kernel=configs["kernels"], beta=configs["beta"], M=configs["mc_samples"], sigma=.5,
                    lamda=args.lamda, length_scale=configs["length_scale"], p=15)

    # Train the decoupled local and global representation learning modules
    if args.train:
        if is_continue:
            rep_model.load_weights('./ckpt/glr_%s_lambda%.1f' %(args.data, args.lamda))
        train_glr(rep_model, trainset, validset, lr=lr, n_epochs=n_epochs, data=args.data)

    # Plot summary performance graphs for the learning framework,
    # including the representation distribution and signal reconstruction plots
    rep_model.load_weights('./ckpt/glr_%s_lambda%.1f' %(args.data, args.lamda))
    plot_reps(testset, rep_model, args.data)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='air_quality', help="dataset to use")
    parser.add_argument('--lamda', type=float, default=1., help="regularization weight")
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    main(args)
