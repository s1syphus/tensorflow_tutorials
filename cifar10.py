"""
This will contain the model data
"""

import argparse
import tensorflow as tf
import cifar10_input
import cifar10_train


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", "--model", help="Model to train or evaluate")
    ap.add_argument("-T", "--train", default=False, help="Train model")
    ap.add_argument("-e", "--evaluate", default=False, help="Evaluate model")
    ap.add_argument("-d", "--data_dir", default='/tmp/cifar10_data', help="Path to image files")
    ap.add_argument("-m", "--model_name", help="Model name")
    ap.add_argument("-b", "--batch_size", default=128, help="Batch Size")
    ap.add_argument("-n", "--training_steps", default=2000, help="Number of training steps")
    ap.add_argument("-r", "--learning_rate", default=0.01, help="Learning Rate")

    # Add in some online preprocessing stuff at some point
    # eg. random_flip, random_scale, etc
    args = vars(ap.parse_args())

    cifar10_input.maybe_download_and_extract(args["data_dir"])
    cifar10_train.train()
