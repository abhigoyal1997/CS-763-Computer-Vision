import argparse
import os
import numpy as np
import torchfile
import torch

from src.Model import Model
from src.Linear import Linear
from src.ReLU import ReLU
from src.Dropout import Dropout

from src.training import train
# torch.set_default_tensor_type(torch.DoubleTensor)

def createModel(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    model = Model()
    for desc in spec:
        desc = desc.split()
        if desc[0] == 'linear':
            in_features, out_features = int(desc[1]), int(desc[2])
            layer = Linear(in_features, out_features)
        elif desc[0] == 'relu':
            layer = ReLU()
        elif desc[0] == 'dropout':
            layer = Dropout(float(desc[1]), isTrain=True)
        else:
            print(desc[0] + ' layer not implemented!')
        model.addLayer(layer)
    return model

def readHparams(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    hparams = {}
    hparams['batch_size'] = int(spec[0])
    hparams['learning_rate'] = float(spec[1])
    hparams['num_epochs'] = int(spec[2])
    hparams['momentum'] = float(spec[3])
    hparams['verbose'] = int(spec[4])
    return hparams

def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '../../../A3_data'
    parser.add_argument('-modelName', help='Will create a folder with given model name and save the trained model in that folder.')
    parser.add_argument('-modelSpec', help='Path to Model Specification file.', default='./bestModel/model_spec.txt')
    parser.add_argument('-trainSpec', help='Path to Training Hyperparam file.', default='./bestModel/train_spec.txt')
    parser.add_argument('-data', help='Path to training instances.', default=os.path.join(data_dir, 'data.bin'))
    parser.add_argument('-target', help='Path to training labels.', default=os.path.join(data_dir, 'labels.bin'))
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # To create the directory
    modelPath = os.path.join('./', args.modelName)
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)
    
    # Create Model
    model = createModel(args.modelSpec)
    # Create Hparams
    hparams = readHparams(args.trainSpec)

    # Model created, Start loading training data
    images = torchfile.load(args.data)
    labels = torchfile.load(args.target)

    # Reshape to (#instances, -1)
    images = np.reshape(images, (images.shape[0],-1))
    # Scale to [0,1]
    images = images.astype(np.float32)/255.0
    
    train(model, hparams, images, labels)
    # TODO : write the weights of the trained model to the modelName dir
    # TODO : create a file config.txt according to relevant conventions