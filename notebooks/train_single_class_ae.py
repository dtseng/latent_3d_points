#!/usr/bin/env python
# coding: utf-8

# ## This notebook will help you train a vanilla Point-Cloud AE with the basic architecture we used in our paper.
#     (it assumes latent_3d_points is in the PYTHONPATH and the structural losses have been compiled)

# In[1]:


# In[2]:



import sys
sys.path.insert(0, "/home/ubuntu")

import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet,                                         load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud


# In[3]:

# Define Basic Parameters

# In[4]:


top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.
top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.

experiment_name = 'single_class_ae'
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = 'chamfer'                   # Loss to optimize: 'emd' or 'chamfer'
# class_name = raw_input('Give me the class name (e.g. "chair"): ').lower()
class_name = 'chair'


# Load Point-Clouds

# In[5]:


syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id)
all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)


# Load default training parameters (some of which are listed beloq). For more details please print the configuration object.
#
#     'batch_size': 50
#
#     'denoising': False     (# by default AE is not denoising)
#
#     'learning_rate': 0.0005
#
#     'z_rotate': False      (# randomly rotate models of each batch)
#
#     'loss_display_step': 1 (# display loss at end of these many epochs)
#     'saver_step': 10       (# over how many epochs to save neural-network)

# In[6]:


train_params = default_train_params()


# In[7]:


encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))


# In[8]:


conf = Conf(n_input = [1948, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
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
conf.held_out_step = 5   # How often to evaluate/print out loss on
                         # held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))


# If you ran the above lines, you can reload a saved model like this:

# In[9]:


load_pre_trained_ae = True
restore_epoch = 400
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch)


# Build AE Model.

# In[10]:


reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)


# Train the AE (save output to train_stats.txt)

# In[ ]:


buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(all_pc_data, conf, log_file=fout)
fout.close()
