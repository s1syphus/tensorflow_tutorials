"""
Evaluation stuff might not this for now
"""

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10_model
import cifar10_input


def eval_once(
        saver,
        top_k_op,
        checkpoint_dir='/tmp/cifar10_data',
        batch_size=128,
        num_examples=10000
):
    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            global_step = checkpoint.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        coord = tf.train.Coordinator()

        threads = []
        try:
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

            num_iter = int(math.ceil(num_examples / batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * batch_size
            step = 0

            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: step %s: precision @ 1 = %.3f' % (datetime.now(), global_step, precision))

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate(eval_data, data_dir='/tmp/cifar10_data/cifar-10-batches-bin', moving_average_decay=0.9999):
    with tf.Graph().as_default() as g:
        eval_data = eval_data == 'test'
        images, labels = cifar10_input.inputs(eval_data=eval_data, data_dir=data_dir)
        logits = cifar10_model.inference(images)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.summary.merge_all()

        eval_once(saver, top_k_op)
