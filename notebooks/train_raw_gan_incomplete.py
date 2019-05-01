#!/usr/bin/env python
# coding: utf-8

# ## This notebook will help you train a raw Point-Cloud GAN.
#
# (Assumes latent_3d_points is in the PYTHONPATH and that a trained AE model exists)

# In[2]:
import logging
from imp import reload
import h2o
import sys
sys.path.insert(0, "/home")
import numpy as np
import os.path as osp
import matplotlib.pylab as plt

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.neural_net import MODEL_SAVER_ID

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet,\
        load_all_point_clouds_under_folder, pickle_data, unpickle_data

from latent_3d_points.src.general_utils import plot_3d_point_cloud
from latent_3d_points.src.tf_utils import reset_tf_graph

from latent_3d_points.src.vanilla_gan import Vanilla_GAN
from latent_3d_points.src.w_gan_gp import W_GAN_GP
from latent_3d_points.src.generators_discriminators import point_cloud_generator,mlp_discriminator, leaky_relu, conditional_point_cloud_generator, conditional_missing_points_generator

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('exp_name', type=str)
parser.add_argument('--reconstr_param', type=float, default=5000)
parser.add_argument('--disc_param', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0001)
args = parser.parse_args()

reload(logging)
logging.basicConfig(level=logging.INFO, filename="{}.txt".format(args.exp_name), filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
logging.info("Starting log. ")
top_out_dir = '../data/'

# Top-dir of where point-clouds are stored.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/'

experiment_name = args.exp_name

n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'
# class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = 'chair'


# In[5]:
logging.info('Reconstr param: {}'.format(args.reconstr_param))
logging.info('Disc param: {}'.format(args.disc_param))

# all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
train_pkl = unpickle_data('/home/latent_3d_points/data/missing_points_dataset/train_data.pkl')
train_data = next(train_pkl)

val_pkl = unpickle_data('/home/latent_3d_points/data/missing_points_dataset/val_data.pkl')
val_data = next(val_pkl)

print 'Shape of DATA =', train_data.point_clouds.shape


# Set GAN parameters.

# In[14]:


use_wgan = True     # Wasserstein with gradient penalty, or not?
n_epochs = args.epochs       # Epochs to train.

plot_train_curve = True
save_gan_model = True
saver_step = np.hstack([np.array([1, 5, 10]), np.arange(50, n_epochs + 1, 50)])

# If true, every 'saver_step' epochs we produce & save synthetic pointclouds.
save_synthetic_samples = True
# How many synthetic samples to produce at each save step.
n_syn_samples = train_data.num_examples

# Optimization parameters
init_lr = args.lr
batch_size = 50
noise_params = {'mu':0, 'sigma': 0.2}
# noise_dim = 128
noise_dim = 1948 # incomplete shape: 2048 - 100 - 1948
beta = 0.5 # ADAM's momentum.

n_out = [n_pc_points, 3] # Dimensionality of generated samples.

discriminator = mlp_discriminator
# generator = point_cloud_generator
train_params = default_train_params() # not actually used
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)


conf = Conf(n_input = [1948, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'], # not actually used
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = top_out_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args,
            n_output = [2048, 3]
           )
conf.experiment_name = experiment_name
generator = conditional_missing_points_generator

if save_synthetic_samples:
    synthetic_data_out_dir = osp.join(top_out_dir, 'OUT/synthetic_samples/', experiment_name)
    create_dir(synthetic_data_out_dir)

if save_gan_model:
    train_dir = osp.join(top_out_dir, 'OUT/raw_gan', experiment_name)
    create_dir(train_dir)

# In[15]:

reset_tf_graph()

if use_wgan:
    lam = 10
    disc_kwargs = {'b_norm': False}
    gan = W_GAN_GP(experiment_name, init_lr, lam, n_out, noise_dim,
                    discriminator, generator, conf,
                    disc_kwargs=disc_kwargs, beta=beta, reconstr_param=args.reconstr_param, disc_param=args.disc_param)

else:
    leak = 0.2
    disc_kwargs = {'non_linearity': leaky_relu(leak), 'b_norm': False}
    gan = Vanilla_GAN(experiment_name, init_lr, n_out, noise_dim,
                      discriminator, generator, beta=beta, disc_kwargs=disc_kwargs)

accum_syn_data = []
train_stats = []


# In[ ]:


# Train the GAN.
print("starting training..")

for _ in range(n_epochs):
    loss, duration = gan._single_epoch_train(train_data, batch_size, noise_params)
    epoch = int(gan.sess.run(gan.increment_epoch))
    print epoch, loss
    logging.info("Epoch: {}, loss: {}".format(epoch, loss))

    if save_gan_model and epoch in saver_step:
        checkpoint_path = osp.join(train_dir, MODEL_SAVER_ID)
        gan.saver.save(gan.sess, checkpoint_path, global_step=gan.epoch)

    if save_synthetic_samples and epoch in saver_step:
        batch_i, _, _ = val_data.next_batch(10)
        inc, real = batch_i[:, :1948, :], batch_i[:, 1948:, :]
        syn_data = gan.generate(inc)

        np.savez(osp.join(synthetic_data_out_dir, 'epoch_' + str(epoch)), syn_data)
        # for k in range(3):  # plot three (synthetic) random examples.
        #     plot_3d_point_cloud(syn_data[k][:, 0], syn_data[k][:, 1], syn_data[k][:, 2],
        #                        in_u_sphere=True)

    train_stats.append((epoch, ) + loss)


# In[27]:


# if plot_train_curve:
#     x = range(len(train_stats))
#     d_loss = [t[1] for t in train_stats]
#     g_loss = [t[2] for t in train_stats]
#     plt.plot(x, d_loss, '--')
#     plt.plot(x, g_loss)
#     plt.title('GAN training. (%s)' %(class_name))
#     plt.legend(['Discriminator', 'Generator'], loc=0)
#
#     plt.tick_params(axis='x', which='both', bottom='off', top='off')
#     plt.tick_params(axis='y', which='both', left='off', right='off')
#
#     plt.xlabel('Epochs.')
#     plt.ylabel('Loss.')
