import argparse
import os
import torchfile
import torch

from src.testing import test
from src.Model import Model
from src.Linear import Linear
from src.ReLU import ReLU
from src.Dropout import Dropout
torch.set_default_tensor_type(torch.DoubleTensor)


def getModel(config_file):
    with open(config_file,'r') as f:
        config = f.readlines()
    weights = torch.load(config[-2].strip())
    biases = torch.load(config[-1].strip())

    model = Model()
    il = 0
    for desc in config[1:-2]:
        desc = desc.split()
        if desc[0] == 'linear':
            in_features, out_features = int(desc[1]), int(desc[2])
            layer = Linear(in_features, out_features)
            layer.W = torch.Tensor(weights[il])
            layer.B = torch.Tensor(biases[il]).view(out_features, 1)
            il += 1
        elif desc[0] == 'relu':
            layer = ReLU()
        elif desc[0] == 'dropout':
            layer = Dropout(float(desc[1]), isTrain=False)
        else:
            print(desc[0] + ' layer not implemented!')
        model.addLayer(layer)

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '../Test'
    parser.add_argument('-modelName', help='Will use this to load Model.')
    parser.add_argument('-data', help='Path to testing instances.', default=os.path.join(data_dir, 'test.bin'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # To load the model from the directory
    configPath = os.path.join('./' + args.modelName, 'config.txt')

    # Get Model (using checkModel.py)
    model = getModel(configPath)
    print('Model initialized!')

    print('Loading data...')
    # Model created, Start loading testing data
    images = torchfile.load(args.data)

    # Reshape to (#instances, -1) and Scale to [0,1]
    images = torch.Tensor(torchfile.load(args.data))
    images = images.view(images.size(0), -1)/255.0

    print('Predicting labels...')
    predictions = test(model, images)
    torch.save(predictions, 'testPrediction.bin')
    print('Predictions saved to testPrediction.bin')
