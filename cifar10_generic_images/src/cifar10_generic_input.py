"""
CIFAR10 Generic Input
"""

import tensorflow as tf
import multiprocessing
from glob import glob


IMAGE_SIZE = 24


def read_cifar10(filename_queue):
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    images = tf.image.decode_jpeg(value, channels=3)

    return images


def distorted_inputs(batch_size=128, data_dir='../test_data/', num_examples_per_epoch_for_train=50000):

    # Get all filenames here
    filenames = glob(data_dir + '*.png')
    print(filenames)
    filename_queue = tf.train.string_input_producer(filenames)
    print(filename_queue)
    images = read_cifar10(filename_queue)
    print(images)

    return

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)


    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(distorted_image)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch_for_train *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    # Get the number of cpus (virtual) available
    num_preprocess_threads = multiprocessing.cpu_count()
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, tf.reshape(label_batch, [batch_size])

