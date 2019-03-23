import argparse
import os
import torchfile
import torch
import numpy as np

from src.Model import Model
from src.RNN import RNN

from src.read_data import get_data
from src.read_labels import get_labels

from src.training import train
torch.set_default_tensor_type(torch.DoubleTensor)  # As asked in assignment to use double tensor
RANDOM_SEED = 12345


def createModel(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    model = Model()
    num_layers = 0
    for desc in spec:
        desc = desc.split()
        if desc[0] == 'rnn':
            in_features, hidden_features, out_features = int(desc[1]), int(desc[2]), int(desc[3])
            layer = RNN(in_features, hidden_features, out_features)
            num_layers += 1
        else:
            print(desc[0] + ' layer not implemented!')
        model.addLayer(layer)
    return model, (spec, num_layers)


def readHparams(spec_file):
    with open(spec_file,'r') as f:
        spec = f.readlines()
    param_keys = [
        'batch_size',
        'learning_rate',
        'learning_rate_decay',
        'num_epochs',
        'momentum',
        'verbose',
        'train_ratio'
    ]
    hparams = {}
    for i in range(len(param_keys)):
        hparams[param_keys[i]] = float(spec[i])
    return hparams


def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '../../A4_data'
    parser.add_argument('-modelName', help='Will create a folder with given model name and save the trained model in that folder.', required=True)
    parser.add_argument('-modelSpec', help='Path to Model Specification file.', default='./bestModel/model_spec.txt')
    parser.add_argument('-trainSpec', help='Path to Training Hyperparam file.', default='./bestModel/train_spec.txt')
    parser.add_argument('-data', help='Path to training instances.', default=os.path.join(data_dir, 'train_data.txt'))
    parser.add_argument('-target', help='Path to training labels.', default=os.path.join(data_dir, 'train_labels.txt'))
    parser.add_argument('-s', dest='train_size', help='Train size', default=None, type=int)
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

    # Model created, Start loading training data
    print('Loading data...')
    data, lengths = get_data(args.data, limit=args.train_size)
    labels = get_labels(args.target, limit=args.train_size)

    print('Training model...')
    train(model, hparams, data, lengths, labels, model_path, model_config, log_interval=1)