"""
This will contain the model data
"""

import argparse
import image_processing
import image_conversion
import multiprocessing
from cifar10_dataset import CIFAR10Data
import tensorflow as tf


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--convert_images", default=False, help="Convert Images")
    ap.add_argument("--num_classes", default=10, help="Number of classes")
    ap.add_argument("--train_images", default='/development/test_data/', help="Path to training image files")
    ap.add_argument("--train_labels", default='/development/data/trainLabels.csv',
                    help="Path to training image labels")
    ap.add_argument("--records_directory", default='/development/data/sharded_data/', help="Records Directory")
    ap.add_argument("--num_preprocess_threads",
                    # default=multiprocessing.cpu_count(),
                    default=1,
                    help="Number of preprocessing threads")

    args = vars(ap.parse_args())

    if args["convert_images"]:
        image_conversion.convert_dir_with_labels_file(
            args["train_images"],
            args["train_labels"],
            args["num_classes"],
            args["records_directory"]
        )

    dataset = CIFAR10Data(subset='train', data_dir=args["records_directory"])
    # This will be moved to the train file soon
    image_processing.inputs(
        dataset=dataset,
        num_preprocess_threads=int(args["num_preprocess_threads"]),
        batch_size=128
    )






