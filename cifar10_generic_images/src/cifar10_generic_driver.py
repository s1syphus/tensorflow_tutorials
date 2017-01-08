"""
This will contain the model data
"""

import argparse
import cifar10_generic_input


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", "--model", help="Model to train or evaluate")
    ap.add_argument("-T", "--train", default=False, help="Train model")
    ap.add_argument("-e", "--evaluate", default=False, help="Evaluate model")
    ap.add_argument("-d", "--data_dir", default='data', help="Path to image files")
    ap.add_argument("-m", "--model_name", help="Model name")
    ap.add_argument("-b", "--batch_size", default=128, help="Batch Size")
    ap.add_argument("-n", "--training_steps", default=5000, help="Number of training steps")
    ap.add_argument("-r", "--learning_rate", default=0.01, help="Learning Rate")

    args = vars(ap.parse_args())

    cifar10_generic_input.distorted_inputs()

    if args["train"]:
        print("Training...")

    if args["evaluate"]:
        print("Time to Evaluate")




