import argparse
import os
import numpy as np
import torchfile
import torch

from src.Model import Model
from src.Linear import Linear
from src.ReLU import ReLU
from src.Dropout import Dropout

from checkModel import getModel
# torch.set_default_tensor_type(torch.DoubleTensor)

def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '../../../A3_data'
    parser.add_argument('-modelName', help='Will use this to load Model.')
    parser.add_argument('-data', help='Path to testing instances.', default=os.path.join(data_dir, 'test.bin'))
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # To load the model from the directory
    configPath = os.path.join('./' + args.modelName, 'config.txt')
    
    # Get Model (using checkModel.py)
    model = getModel(configPath)

    # Model created, Start loading testing data
    images = torchfile.load(args.data)

    # Reshape to (#instances, -1)
    images = np.reshape(images, (images.shape[0],-1))
    # Scale to [0,1]
    images = images.astype(np.float32)/255.0
    
    predictions = test(model, images)
    # TODO : write predictions (numpy array) into a file