import argparse
import os
import torchfile
import torch

from src.Model import Model
from src.RNN import RNN

from src.read_data import get_data 

from src.testing import test
torch.set_default_tensor_type(torch.DoubleTensor)


def getModel(config_file):
    with open(config_file,'r') as f:
        config = f.readlines()
    Whh = torch.load(config[-5].strip())
    Wxh = torch.load(config[-4].strip())
    Why = torch.load(config[-3].strip())
    Bhh = torch.load(config[-2].strip())
    Bhy = torch.load(config[-1].strip())

    model = Model()
    il = 0
    for desc in config[1:-5]:
        desc = desc.split()
        if desc[0] == 'rnn':
            in_features, hidden_features, out_features = int(desc[1]), int(desc[2]), int(desc[3])
            layer = RNN(in_features, hidden_features, out_features)
            layer.Whh = torch.Tensor(Whh[il])
            layer.Wxh = torch.Tensor(Wxh[il])
            layer.Why = torch.Tensor(Why[il])
            layer.Bhh = torch.Tensor(Bhh[il]).view(hidden_features, 1)
            layer.Bhy = torch.Tensor(Bhy[il]).view(out_features, 1)
            il += 1
        else:
            print(desc[0] + ' layer not implemented!')
        model.addLayer(layer)

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    data_dir = '../../A4_data'
    parser.add_argument('-modelName', help='Will use this to load Model.')
    parser.add_argument('-data', help='Path to testing instances.', default=os.path.join(data_dir, 'test_data.txt'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # To load the model from the directory
    configPath = os.path.join('./' + args.modelName, 'config.txt')

    model = getModel(configPath)
    print('Model initialized!')

    print('Loading data...')
    # Model created, Start loading testing data
    data, lengths = get_data(args.data)

    print('Predicting labels...')
    predictions = test(model, data, lengths)
    torch.save(predictions, 'testPrediction.bin')
    print('Predictions saved to testPrediction.bin')