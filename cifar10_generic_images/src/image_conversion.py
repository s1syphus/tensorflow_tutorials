"""
Convert Images

Make this much more efficient

although this only needs to be run once so is a lower priority

"""

import tensorflow as tf
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
import os


def convert_dir_with_folder_labels(data_dir):
    pass


def convert_dir_no_labels(data_dir):
    pass


def convert_dir_with_labels_file(data_dir, labels_file_name, record_name, records_directory, label_encoder=None):
    """
     Eventually add in a check for labels file

     Make this more efficient at some point, just trying to get something working
    """
    file_labels = pd.read_csv(labels_file_name)
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder = label_encoder.fit(file_labels['label'])

    input_files = glob(data_dir + '*.png')
    writer = tf.python_io.TFRecordWriter(records_directory + 'record_1.tfrecord')

    height = 32
    width = 32
    channels = 3

    for filename in input_files:
        # print(filename)
        index = int(filename.split('/')[-1].split('.')[0])
        image = Image.open(filename)
        image = np.asarray(image, np.uint8)
        idx = file_labels[file_labels['id'] == index].index[0]
        human_label = file_labels.get_value(idx, 'label')
        label = int(label_encoder.transform([human_label]))
        example = _convert_to_example(
            filename=filename,
            image_buffer=image.tostring(),
            label=label,
            text=human_label,
            height=height,
            width=width,
            channels=channels
        )
        writer.write(example.SerializeToString())


def _convert_to_example(filename, image_buffer, label, text, height, width, channels):
    colorspace = 'RGB'
    image_format = 'PNG'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
        'image/channels': _int64_feature(channels),
        'image/class/label': _int64_feature(label),
        'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
        'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
        'image/filename': _bytes_feature(tf.compat.as_bytes(os.path.basename(filename))),
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
