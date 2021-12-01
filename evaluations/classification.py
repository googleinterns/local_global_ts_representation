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

from absl import flags
from absl import app
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
sns.set();
import tensorflow as tf

from gl_rep.models import EncoderLocal, EncoderGlobal, WindowDecoder
from gl_rep.glr import GLR
from gl_rep.data_loaders import airq_data_loader, simulation_loader, physionet_data_loader, har_data_loader
from baselines.gpvae import GPVAE, Decoder
from baselines.vae import VAE, Encoder, Decoder


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)

FLAGS = flags.FLAGS


# Physionet config
flags.DEFINE_integer('window_size', 4, 'Window size for the local encoder')
flags.DEFINE_integer('rep_size', 8, 'Size of the latent representation vector')
flags.DEFINE_float('global_sample_len', 0.6, 'ratio of the signal to look at')
flags.DEFINE_float('lamda', 0.0, 'Regularization weight in GLR')
flags.DEFINE_float('lr', 1e-3, 'Learning rate')
flags.DEFINE_integer('repeat', 3, 'Number of random windows sampled from each data sample during the training')
flags.DEFINE_integer('n_epochs', 60, 'Number of training epochs')
flags.DEFINE_integer('n_classes', 4, 'Number of classification classes')
flags.DEFINE_string('mode', 'glr', 'Which encoder baseline to use')
flags.DEFINE_string('data', 'physionet', 'Which dataset to use')
flags.DEFINE_string('category', 'global', 'Which representations to use for the classification')
flags.DEFINE_string('experiment', 'classification', 'Which experiment setup to run')


# air quality config
# flags.DEFINE_integer('window_size', 24, 'Window size for the local encoder')
# flags.DEFINE_integer('rep_size', 8, 'Size of the latent representation vector')
# flags.DEFINE_float('global_sample_len', 0.3, 'ratio of the signal to look at')
# flags.DEFINE_float('lamda', 0.0, 'Regularization weight in GLR')
# flags.DEFINE_float('lr', 1e-3, 'Learning rate')
# flags.DEFINE_integer('repeat', 5, 'Number of random windows sampled from each data sample during the training')
# flags.DEFINE_integer('n_epochs', 80, 'Number of training epochs')
# flags.DEFINE_integer('n_classes', 12, 'Number of classification classes')
# flags.DEFINE_string('mode', 'glr', 'Which encoder baseline to use')
# flags.DEFINE_string('data', 'air_quality', 'Which dataset to use')
# flags.DEFINE_string('category', 'global', 'Which representations to use for the classification')
# flags.DEFINE_string('experiment', 'classification', 'Which experiment setup to run')

# har data config
# flags.DEFINE_integer('window_size', 5, 'Window size for the local encoder')
# flags.DEFINE_integer('rep_size', 8, 'Size of the latent representation vector')
# flags.DEFINE_float('global_sample_len', 0.3, 'ratio of the signal to look at')
# flags.DEFINE_float('lamda', 0.5, 'Regularization weight in GLR')
# flags.DEFINE_float('lr', 1e-3, 'Learning rate')
# flags.DEFINE_integer('repeat', 5, 'Number of random windows sampled from each data sample during the training')
# flags.DEFINE_integer('n_epochs', 200, 'Number of training epochs')
# flags.DEFINE_integer('n_classes', 6, 'Number of classification classes')
# flags.DEFINE_string('mode', 'e2e', 'Which encoder baseline to use')
# flags.DEFINE_string('data', 'har', 'Which dataset to use')
# flags.DEFINE_string('category', 'local', 'Which representations to use for the classification')
# flags.DEFINE_string('experiment', 'classification', 'Which experiment setup to run')



def main(_argv):
    with open('configs.json') as config_file:
        configs = json.load(config_file)[FLAGS.data]
    test_accuracies, test_losses = [], []
    for cv in range(2):
        # Load the dataset
        if FLAGS.data=='air_quality':
            trainset, validset, testset, normalization_specs = airq_data_loader(normalize='mean_zero')
        elif FLAGS.data=='physionet':
            trainset, validset, testset, normalization_specs = physionet_data_loader(normalize='mean_zero')
        elif FLAGS.data=='har':
            trainset, validset, testset, normalization_specs = har_data_loader(normalize='none')

        # Create the representation learning models
        if FLAGS.mode=='gpvae':
            file_name = 'gpvae%d_%s' %(FLAGS.rep_size, FLAGS.data)
            encoder = EncoderLocal(zl_size=FLAGS.rep_size, hidden_sizes=configs["baseline_encoder_size"])
            decoder = Decoder(output_size=configs["feature_size"], output_length=configs["window_size"],
                              hidden_sizes=configs["baseline_decoder_size"])
            rep_model = GPVAE(encoder, decoder, time_length=configs["t_len"], data_dim=configs["feature_size"],
                              window_size=configs["window_size"], kernel=['cauchy'], beta=1., M=1,
                              sigma=1.0, length_scale=1.0, kernel_scales=1)
            rep_model.load_weights('./ckpt/%s' % file_name)
            representation_classifier = LinearClassifier(rep_model.local_encoder, n_classes=FLAGS.n_classes)
        elif FLAGS.mode=='vae':
            file_name = 'vae%d_%s' %(FLAGS.rep_size, FLAGS.data)
            encoder = Encoder(zl_size=FLAGS.rep_size, hidden_sizes=configs["baseline_encoder_size"])
            decoder = Decoder(output_size=configs["feature_size"], output_length=configs["window_size"],
                              hidden_sizes=configs["baseline_decoder_size"])
            rep_model = VAE(encoder=encoder, decoder=decoder, data_dim=configs["feature_size"],
                            M=configs["mc_samples"], beta=1., sample_len=configs["t_len"])
            rep_model.load_weights('./ckpt/%s' % file_name)
            representation_classifier = LinearClassifier(rep_model.encoder, n_classes=FLAGS.n_classes)
        elif FLAGS.mode == 'glr':
            file_name = 'glr_%s_lambda%.1f' % (FLAGS.data, FLAGS.lamda)
            zt_encoder = EncoderLocal(zl_size=configs["zl_size"], hidden_sizes=configs["glr_local_encoder_size"])
            zg_encoder = EncoderGlobal(zg_size=configs["zg_size"], hidden_sizes=configs["glr_global_encoder_size"])
            dec = WindowDecoder(output_size=configs["feature_size"],
                                output_length=configs["window_size"],
                                hidden_sizes=configs["glr_decoder_size"])
            rep_model = GLR(global_encoder=zg_encoder, local_encoder=zt_encoder, decoder=dec,
                            window_size=configs["window_size"], time_length=configs["t_len"],
                            data_dim=configs["feature_size"], kernel_scales=configs["kernel_scales"],
                            kernel=configs["kernels"], beta=0.1, M=configs["mc_samples"], sigma=.5,
                            lamda=FLAGS.lamda, length_scale=configs["length_scale"], p=15)
            rep_model.load_weights('./ckpt/%s' % file_name)
            if FLAGS.category=='global':
                representation_classifier = LinearClassifier(rep_model.global_encoder, n_classes=FLAGS.n_classes)
            elif FLAGS.category=='local':
                representation_classifier = LinearClassifier(rep_model.local_encoder, n_classes=FLAGS.n_classes)
        elif FLAGS.mode=='e2e':
            file_name = 'e2e_%s' % (FLAGS.data)
            if FLAGS.category=='global':
                representation_classifier = E2E_model(encoder_model=EncoderGlobal(zg_size=FLAGS.rep_size,
                                                                                  hidden_sizes=configs["baseline_encoder_size"]),
                                                      n_classes=FLAGS.n_classes)
            elif FLAGS.category=='local':
                representation_classifier = E2E_model(encoder_model=EncoderLocal(zl_size=FLAGS.rep_size,
                                                                                 hidden_sizes=configs["baseline_encoder_size"]),
                                                      n_classes=FLAGS.n_classes)

        test_acc, test_loss = classification_exp(representation_classifier, FLAGS.mode=='e2e', configs["feature_size"],
                                                 file_name, data=FLAGS.data, datasets=(trainset, validset, testset))
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
    print('\n Overall Test Performance ==========> Accuracy = %.2f +- %.2f \t Loss = %.2f +- %.2f' % (
        100 * np.mean(test_accuracies), 100 * np.std(test_accuracies), np.mean(test_losses), np.std(test_losses)))

            

def classification_exp(representation_classifier, e2e, feature_size, file_name, data, datasets):
    """Run the classification experiment"""
    trainset, validset, testset = datasets[0], datasets[1], datasets[2]
    ckpt_name = './ckpt/representation_classifier_%s' % (file_name)

    print('************************* TRAINING %s CLASSIFIER *************************'%FLAGS.mode)
    train_classifier(trainset, validset, model=representation_classifier, n_epochs=FLAGS.n_epochs-20, lr=FLAGS.lr, ckpt_name=ckpt_name,
                     data=data, file_name=file_name, feature_size=feature_size, e2e=e2e)
    test_loss, test_acc = run_classifier_epoch(representation_classifier, testset, data=data, train=False, is_print=True, repeat=FLAGS.repeat)
    print('Testset ==========> Accuracy = %.3f\n'%test_acc)
    return test_acc, test_loss


class E2E_model(tf.keras.Model):
    """An end to end model composed of an encoder and a classifier for identifying the subgroups of the data

    Attributes:
        encoder_model: An encoder to learn the representation of a time series over time
        n_classes: Number of subgroups of data
    """
    def __init__(self, encoder_model, n_classes, regression=False):
        super(E2E_model, self).__init__()
        self.encoder = encoder_model
        self.n_classes = n_classes
        if regression:
            self.classifier = tf.keras.Sequential([tf.keras.layers.Dense(2*self.n_classes, dtype=tf.float32, activation=tf.nn.relu),
                                                   tf.keras.layers.Dense(1, dtype=tf.float32, activation=tf.nn.rel)])
        else:
            self.classifier = tf.keras.Sequential([tf.keras.layers.Dense(2*self.n_classes, dtype=tf.float32, activation=tf.nn.relu),
                                                   tf.keras.layers.Dense(self.n_classes, dtype=tf.float32, activation=tf.nn.softmax)])

    def __call__(self, x, mask=None, is_train=False, rnd_t=0, window_size=None):
        if FLAGS.category=='global':
            sample_len = int(x.shape[1] * FLAGS.global_sample_len)
            encodings = self.encoder(x[:, rnd_t:sample_len+ rnd_t, :],
                                     None if mask is None else mask[:, rnd_t:sample_len + rnd_t, :]).mean()
        elif FLAGS.category=='local':
            encodings = self.encoder(x[:, rnd_t*window_size:(rnd_t+1)*window_size, :],
                                     None if mask is None else mask[:, rnd_t*window_size:(rnd_t+1)*window_size, :],
                                     window_size)
            encodings = tf.transpose(encodings.sample(), perm=(0, 2, 1))[:, 0, :]
        probs = self.classifier(encodings)
        return probs


class LinearClassifier(tf.keras.Model):
    def __init__(self, encoder, n_classes, regression=False):
        """
        End-to-end supervised classifier for the global representations
        """
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        if regression:
            self.n_classes = 1
            self.classifier = tf.keras.Sequential([tf.keras.layers.Dense(self.n_classes, dtype=tf.float32, activation=tf.nn.relu),
                                                   tf.keras.layers.Dense(1, dtype=tf.float32, activation=tf.nn.relu)])
        else:
            self.n_classes = n_classes
            self.classifier = tf.keras.Sequential([tf.keras.layers.Dense(self.n_classes, activation=tf.nn.relu),
                                                   tf.keras.layers.Dense(self.n_classes)])#, activation=tf.nn.softmax)])

    def __call__(self, x, mask=None, window_size=None, rnd_t=0, is_train=False):
        if FLAGS.mode == 'gpvae':
            if not mask is None:
                mask = mask[:, rnd_t*window_size:(rnd_t+1)*window_size, :]
            pz_t = self.encoder(x[:, rnd_t*window_size:(rnd_t+1)*window_size, :], mask, window_size=window_size)
            z_t = tf.transpose(pz_t.sample(), perm=(0, 2, 1))
            encodings = z_t[:,0,:]
        elif FLAGS.mode == 'vae':
            t = np.random.randint(0, x.shape[1] - window_size)
            if not mask is None:
                mask = mask[:, t:t + window_size, :]
            pz_t = self.encoder(x[:, t:t + window_size, :], mask, window_size=window_size)
            z_t = pz_t.sample()
            encodings = z_t[:, 0, :]
        else:
            if FLAGS.category=='global':
                sample_len = int(x.shape[1] * FLAGS.global_sample_len)
                encodings = self.encoder(x[:, rnd_t:sample_len + rnd_t, :],
                                         None if mask is None else mask[:, rnd_t:sample_len + rnd_t, :]).mean()
            elif FLAGS.category=='local':
                encodings = self.encoder(x[:, rnd_t*FLAGS.window_size:(rnd_t+1)*window_size, :],
                                         None if mask is None else mask[:, rnd_t*window_size:(rnd_t+1)*window_size, :],
                                         window_size)
                encodings = tf.transpose(encodings.sample(), perm=(0, 2, 1))[:,0,:]
        probs = self.classifier(encodings)
        return probs


def train_classifier(trainset, validset, model, n_epochs, lr, ckpt_name,  file_name, feature_size, data, e2e=True):
    """Train a classifier to classify the subgroup of time series"""
    losses_train, losses_val, acc_train, acc_val = [], [], [], []
    optimizer = tf.keras.optimizers.Adam(lr)
    model(x=tf.random.normal(shape=(2, 1000, feature_size), dtype=tf.float32),
          mask=tf.zeros(shape=(2, 1000, feature_size), dtype=tf.float32),
          window_size=FLAGS.window_size)
    if e2e:
        trainable_var = model.trainable_variables
        plt_name = './plots/evaluations/%s_classifier.pdf' % (file_name)
    else:
        trainable_var = model.classifier.trainable_variables
        plt_name = './plots/evaluations/%s_classifier.pdf' % (file_name)
    for epoch in range(n_epochs+1):
        run_classifier_epoch(model, trainset, data=data, optimizer=optimizer, train=True ,
                             trainable_vars=trainable_var, repeat=FLAGS.repeat)
        valid_loss, valid_acc = run_classifier_epoch(model, validset, data=data, train=False, repeat=FLAGS.repeat)
        train_loss, train_acc = run_classifier_epoch(model, trainset, data=data, train=False, repeat=FLAGS.repeat)
        losses_train.append(train_loss)
        acc_train.append(train_acc)
        losses_val.append(valid_loss)
        acc_val.append(valid_acc)
        if epoch%5==0:
            print('=' * 30)
            print('Epoch %d' % epoch, '(Learning rate: %.5f)' % (lr))
            print("Training loss = %.3f \t Training accuracy = %.3f" % (train_loss, train_acc))
            print("Validation loss = %.3f \t Validation accuracy = %.3f" % (valid_loss, valid_acc))
            model.save_weights(ckpt_name)
    # Plot losses
    if not os.path.exists('./plots/evaluations'):
        os.mkdir('./plots/evaluations')
    plt.figure()
    plt.plot(losses_train, label='Train loss')
    plt.plot(losses_val, label='Validation loss')
    plt.legend()
    plt.savefig(plt_name)


def run_classifier_epoch(model, dataset, data, optimizer=None, train=False , trainable_vars=None, repeat=5, is_print=False):
    """Training epoch of a classifier"""
    ce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    epoch_loss, epoch_acc= [], []
    for _ in range(repeat):
        for i, batch in dataset.enumerate():
            x_seq, mask_seq, x_lens = batch[0], batch[1], batch[2]
            mask_seq = tf.cast(mask_seq, tf.float32)
            if FLAGS.category=='global':
                if FLAGS.mode=='glr':
                    sample_len = int(x_seq.shape[1] * FLAGS.global_sample_len)
                    rnd_t = np.random.randint(0, max((min(x_lens) - sample_len), 1))
                else:
                    rnd_t = np.random.randint(0, (
                                (x_seq.shape[1]// FLAGS.window_size if x_lens is None else min(x_lens)) // FLAGS.window_size) - 1)
                if data=='air_quality':
                    labels = tf.convert_to_tensor([int(m.numpy().decode("utf-8"))-1 for m in batch[4][:, 1]], dtype=tf.int64)
                elif data=='har':
                    labels = batch[4]
                elif data=='physionet':
                    labels = batch[4][:, 3] - 1
                    labels = tf.cast(labels, tf.int64)
                labels_one_hot = tf.one_hot(labels, model.n_classes)

            elif FLAGS.category=='local':
                rnd_t = np.random.randint(0, ((x_seq.shape[1] if x_lens is None else min(x_lens))//FLAGS.window_size)-1)
                if data == 'har':
                    all_labels = batch[3]-1
                    labels = tf.math.argmax(tf.math.bincount(all_labels[:, rnd_t * FLAGS.window_size:(rnd_t + 1) * FLAGS.window_size], axis=-1), axis=-1)
                    labels_one_hot = tf.one_hot(labels, model.n_classes)

            if train:
                with tf.GradientTape() as tape:
                    predictions = model(x_seq, mask_seq, rnd_t=rnd_t, window_size=FLAGS.window_size)
                    loss = ce_loss(labels_one_hot, predictions)
                gradients = tape.gradient(loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients, trainable_vars))
            else:
                predictions = model(x_seq, mask_seq, rnd_t=rnd_t, window_size=FLAGS.window_size)
                loss = ce_loss(labels_one_hot, predictions)
            accuracy = tf.cast(labels==tf.math.argmax(predictions, axis=-1), tf.float32)
            accuracy = np.mean(accuracy)
            epoch_loss.append(loss.numpy())
            epoch_acc.append(accuracy)
    return np.mean(epoch_loss), np.mean(epoch_acc)


if __name__=="__main__":
    app.run(main)
