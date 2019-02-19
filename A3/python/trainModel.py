import argparse
import os
import torchfile
from Model import Model
from Linear import Linear
from ReLU import ReLU

# Handle the flags
# Create Model directory
# Input training data -> useful to create generic code that can be used to input testing data as well
# train

def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '../../../A3_data'
    parser.add_argument('-modelName', help='Will create a folder with given model name and save the model in that folder.')
    parser.add_argument('-data', help='Path to training instances.', default=os.path.join(data_dir, 'data.bin'))
    parser.add_argument('-target', help='Path to training labels.', default=os.path.join(data_dir, 'labels.bin'))
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.modelName):
        os.makedirs(args.modelName)
    
    