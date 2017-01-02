"""
Train the CIFAR10 model
"""

import time
import tensorflow as tf
import cifar10_model
import cifar10_input


def train(
        max_steps=5000,
        data_dir='/tmp/cifar10_data/cifar-10-batches-bin',
        batch_size=128
):
    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        # Get images and labels for CIFAR-10.
        images, labels = cifar10_input.distorted_inputs(batch_size=batch_size, data_dir=data_dir)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10_model.inference(images)

        # Calculate loss.
        loss = cifar10_model.loss(logits, labels)

