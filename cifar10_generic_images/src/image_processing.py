"""
Convert Images

Use bounding boxes in the future?

"""

import tensorflow as tf

IMAGE_SIZE = 32


def inputs(dataset, num_preprocess_threads, batch_size):
    with tf.device('/cpu:0'):
        images, labels = batch_inputs(
            dataset,
            batch_size,
            train=False,
            num_preprocess_threads=num_preprocess_threads)

    return images, labels


def batch_inputs(dataset, batch_size, train, num_preprocess_threads):
    with tf.name_scope('batch_processing'):
        data_files = dataset.data_files()
        # Create filename_queue
        if train:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16)
        else:
            filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=1)

        reader = dataset.reader()
        _, example_serialized = reader.read(filename_queue)

        images_and_labels = []
        for thread_id in range(num_preprocess_threads):
            image_buffer, label, _ = parse_example_proto(example_serialized)
            image = image_preprocessing(image_buffer, train, thread_id)
            images_and_labels.append([image, label])

        print(images_and_labels)

        return

        capacity = 2 * num_preprocess_threads * batch_size

        images, label_index_batch = tf.train.batch_join(
            images_and_labels,
            batch_size=batch_size,
            capacity=capacity
        )

        # Reshape images into these desired dimensions.
        height = 32
        width = 32
        depth = 3

        images = tf.cast(images, tf.float32)
        images = tf.reshape(images, shape=[batch_size, height, width, depth])

        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        return images, tf.reshape(label_index_batch, [batch_size])


def parse_example_proto(example_serialized):
    # Dense features in Example proto.
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                            default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                                default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
    }
    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)
    return features['image/encoded'], label, features['image/class/text']


def decode_png(image_buffer, scope=None):
    with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
        image = tf.image.decode_png(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def image_preprocessing(image_buffer, train, thread_id=0):
    image = decode_png(image_buffer)
    # if train:
    #     image = distort_image(image, IMAGE_SIZE, IMAGE_SIZE, thread_id)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image



