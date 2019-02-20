import argparse
import os
import torchfile
import torch
import numpy as np

from src.Model import Model
from src.Linear import Linear
from src.ReLU import ReLU
from src.Dropout import Dropout

from src.training import train
torch.set_default_tensor_type(torch.DoubleTensor)  # As asked in assignment to use double tensor
RANDOM_SEED = 12345


def createModel(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    model = Model()
    num_linear_layers = 0
    for desc in spec:
        desc = desc.split()
        if desc[0] == 'linear':
            in_features, out_features = int(desc[1]), int(desc[2])
            layer = Linear(in_features, out_features)
            num_linear_layers += 1
        elif desc[0] == 'relu':
            layer = ReLU()
        elif desc[0] == 'dropout':
            layer = Dropout(float(desc[1]), isTrain=True)
        else:
            print(desc[0] + ' layer not implemented!')
        model.addLayer(layer)
    return model, (spec, num_linear_layers)


def readHparams(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    hparams = {}
    hparams['batch_size'] = int(spec[0])
    hparams['learning_rate'] = float(spec[1])
    hparams['num_epochs'] = int(spec[2])
    hparams['momentum'] = float(spec[3])
    hparams['verbose'] = int(spec[4])
    hparams['train_ratio'] = float(spec[5])
    return hparams


def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '../Train'
    parser.add_argument('-modelName', help='Will create a folder with given model name and save the trained model in that folder.', required=True)
    parser.add_argument('-modelSpec', help='Path to Model Specification file.', default='./bestModel/model_spec.txt')
    parser.add_argument('-trainSpec', help='Path to Training Hyperparam file.', default='./bestModel/train_spec.txt')
    parser.add_argument('-data', help='Path to training instances.', default=os.path.join(data_dir, 'data.bin'))
    parser.add_argument('-target', help='Path to training labels.', default=os.path.join(data_dir, 'labels.bin'))
    return parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    args = parse_args()

    # To create the directory
    model_path = os.path.join('./', args.modelName)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Create Model
    model, model_config = createModel(args.modelSpec)
    # Create Hparams
    hparams = readHparams(args.trainSpec)
    print('Model initialized!')

    print('Loading data...')
    # train_size = 500

    # Model created, Start loading training data
    images = torch.Tensor(torchfile.load(args.data))
    labels = torch.Tensor(torchfile.load(args.target))

    # Reshape to (#instances, -1) and Scale to [0,1]
    images = images.view(images.size(0), -1)/255.0

    print('Training model...')
    train(model, hparams, images, labels, model_path, model_config)
