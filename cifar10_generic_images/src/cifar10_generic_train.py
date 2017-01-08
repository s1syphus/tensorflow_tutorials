"""
CIFAR10 Generic train


At some point add multi_gpu train stuff


"""

import time
import tensorflow as tf
import cifar10_generic_model
import cifar10_generic_input
from datetime import datetime


def train(
        max_steps=5000,
        data_dir='data/',
        batch_size=128,
        checkpoint_dir='data/',
        log_placement=False
):
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = cifar10_generic_input.distorted_inputs(batch_size=batch_size, data_dir=data_dir)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10_generic_model.inference(images)

        # Calculate loss.
        loss = cifar10_generic_model.loss(logits, labels)

        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = train_step(loss, global_step, batch_size=batch_size)

        # Not a big fan of how this is done. Fix at some point soon
        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""

            def begin(self):
                self._step = -1

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=checkpoint_dir,
                hooks=[tf.train.StopAtStepHook(last_step=max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(
                    log_device_placement=log_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def train_step(
        total_loss,
        global_step,
        num_examples_per_epoch_for_train=50000,
        batch_size=128,
        num_epochs_per_decay=350.0,
        initial_learning_rate=0.1,
        learning_rate_decay_factor=0.1,
        moving_average_decay=0.9999
):
    num_batches_per_epoch = num_examples_per_epoch_for_train / batch_size
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)
    eta = tf.train.exponential_decay(
        initial_learning_rate,
        global_step,
        decay_steps,
        learning_rate_decay_factor,
        staircase=True
    )

    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(eta)
        grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def _add_loss_summaries(total_loss, decay=0.9):
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(decay, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    return loss_averages_op
