'''
Created on May 22, 2018

Author: Achlioptas Panos (Github ID: optas)
'''

import numpy as np
import time
import tensorflow as tf

from tflearn import is_training
from . gan import GAN
from .. external.structural_losses.tf_nndistance import nn_distance
# from .. external.structural_losses.tf_approxmatch import approx_match, match_cost


class W_GAN_GP(GAN):
    '''Gradient Penalty.
    https://arxiv.org/abs/1704.00028
    '''

    def __init__(self, name, learning_rate, lam, n_output, noise_dim, discriminator, generator, configuration, beta=0.5, gen_kwargs={}, disc_kwargs={}, graph=None):
        assert noise_dim == 1948

        GAN.__init__(self, name, graph)

        self.noise_dim = noise_dim
        self.n_output = n_output
        self.discriminator = discriminator
        self.generator = generator

        c = configuration

        with tf.variable_scope(name):
            #  self.noise = tf.placeholder(tf.float32, shape=[None, noise_dim])            # Noise vector.
            self.incomplete_input = tf.placeholder(tf.float32, shape=[None, noise_dim, 3])
            self.real_pc = tf.placeholder(tf.float32, shape=[None] + self.n_output)     # Ground-truth.

            with tf.variable_scope('generator'):
                self.generator_out = self.generator(self.incomplete_input, configuration, **gen_kwargs)

            with tf.variable_scope('discriminator') as scope:
                self.real_prob, self.real_logit = self.discriminator(self.real_pc, scope=scope, **disc_kwargs)
                self.synthetic_prob, self.synthetic_logit = self.discriminator(self.generator_out, reuse=True, scope=scope, **disc_kwargs)

            # Compute WGAN losses
            # discriminator loss
            self.loss_d = tf.reduce_mean(self.synthetic_logit) - tf.reduce_mean(self.real_logit)

            # generator loss
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.generator_out, self.real_pc)
            l2_loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)

            # reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            # w_reg_alpha = 1.0
            # for rl in reg_losses:
            #     self.loss += (w_reg_alpha * rl)
            self.g_reconstr_loss = 10000/2.*l2_loss
            self.g_disc_loss = -0.5*tf.reduce_mean(self.synthetic_logit)

            # self.loss_g = -0.2*self.g_disc_loss + 0.8*self.g_reconstr_loss
            self.loss_g = self.g_disc_loss + self.g_reconstr_loss


            # Compute gradient penalty at interpolated points
            ndims = self.real_pc.get_shape().ndims
            batch_size = tf.shape(self.real_pc)[0]
            alpha = tf.random_uniform(shape=[batch_size] + [1] * (ndims - 1), minval=0., maxval=1.)
            differences = self.generator_out - self.real_pc
            interpolates = self.real_pc + (alpha * differences)

            with tf.variable_scope('discriminator') as scope:
                gradients = tf.gradients(self.discriminator(interpolates, reuse=True, scope=scope, **disc_kwargs)[1], [interpolates])[0]

            # Reduce over all but the first dimension
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=range(1, ndims)))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.loss_d += lam * gradient_penalty

            train_vars = tf.trainable_variables()
            d_params = [v for v in train_vars if v.name.startswith(name + '/discriminator/')]
            g_params = [v for v in train_vars if v.name.startswith(name + '/generator/')]

            self.opt_d = self.optimizer(learning_rate, beta, self.loss_d, d_params)
            self.opt_g = self.optimizer(learning_rate, beta, self.loss_g, g_params)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
            self.init = tf.global_variables_initializer()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def generator_noise_distribution(self, n_samples, ndims, mu, sigma):
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def _single_epoch_train(self, train_data, batch_size, noise_params, discriminator_boost=5):
        '''
        see: http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/
             http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
        '''
        n_examples = train_data.num_examples
        epoch_loss_d = 0.
        epoch_loss_g = 0.
        epoch_g_disc_loss = 0.
        epoch_g_reconstr_loss = 0.
        batch_size = batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        iterations_for_epoch = n_batches / discriminator_boost

        is_training(True, session=self.sess)
        try:
            # Loop over all batches
            for _ in xrange(iterations_for_epoch):
                for _ in range(discriminator_boost):
                    batch_i, _, _ = train_data.next_batch(batch_size)
                    # z is incomplete data, feed is the complete PC
                    z, feed = batch_i[:, :1948, :], batch_i[:, 1948:, :]
                    # z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)

                    feed_dict = {self.real_pc: feed, self.incomplete_input: z}
                    _, loss_d = self.sess.run([self.opt_d, self.loss_d], feed_dict=feed_dict)
                    epoch_loss_d += loss_d

                # Update generator.
                # z = self.generator_noise_distribution(batch_size, self.noise_dim, **noise_params)
                batch_i, _, _ = train_data.next_batch(batch_size)
                # z is incomplete data, feed is the complete PC
                z, feed = batch_i[:, :1948, :], batch_i[:, 1948:, :]

                feed_dict = {self.incomplete_input: z, self.real_pc: feed}
                _, loss_g, g_disc_loss, g_reconstr_loss = self.sess.run([self.opt_g, self.loss_g, self.g_disc_loss, self.g_reconstr_loss], feed_dict=feed_dict)
                epoch_loss_g += loss_g
                epoch_g_disc_loss += g_disc_loss
                epoch_g_reconstr_loss += g_reconstr_loss

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        epoch_loss_d /= (iterations_for_epoch * discriminator_boost)
        epoch_loss_g /= iterations_for_epoch
        epoch_g_disc_loss /= iterations_for_epoch
        epoch_g_reconstr_loss /= iterations_for_epoch

        duration = time.time() - start_time
        return (epoch_loss_d, epoch_loss_g, epoch_g_reconstr_loss, epoch_g_disc_loss), duration
