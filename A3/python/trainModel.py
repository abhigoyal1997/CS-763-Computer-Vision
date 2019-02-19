import argparse
import os
import numpy as np
import torchfile
from src.Model import Model
from src.Linear import Linear
from src.ReLU import ReLU
from src.Dropout import Dropout

def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '../../../A3_data'
    parser.add_argument('-modelName', help='Will create a folder with given model name and save the model in that folder.')
    parser.add_argument('-data', help='Path to training instances.', default=os.path.join(data_dir, 'data.bin'))
    parser.add_argument('-target', help='Path to training labels.', default=os.path.join(data_dir, 'labels.bin'))
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Something to this effect - to create the directory
    if not os.path.exists(args.modelName):
        os.makedirs(args.modelName)
    
    images = torchfile.load(args.data)
    labels = torchfile.load(args.target)

    curr_shape = np.shape(images)
    images = np.reshape(images, (curr_shape[0],-1)) # reshape to (#instances, -1)

    images = images.astype(np.float32) # cast to float
    images = images/255.0 # scale to [0,1]

    