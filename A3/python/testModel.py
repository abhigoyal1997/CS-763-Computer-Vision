import argparse
import os
import torchfile
import torch

from checkModel import getModel
from src.testing import test
torch.set_default_tensor_type(torch.DoubleTensor)


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

    # Model created, Start loading testing data
    images = torchfile.load(args.data)

    # Reshape to (#instances, -1) and Scale to [0,1]
    images = torch.Tensor(torchfile.load(args.data))
    images = images.view(images.size(0), -1)/255.0

    predictions = test(model, images)
    torch.save(predictions, 'testPrediction.bin')
