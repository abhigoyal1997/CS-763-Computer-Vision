import argparse
import torchfile
import torch
from src.Model import Model
from src.Linear import Linear
from src.ReLU import ReLU
from src.Dropout import Dropout
torch.set_default_tensor_type(torch.DoubleTensor)


def getModel(config_file):
    with open(config_file,'r') as f:
        config = f.readlines()
    weights = torchfile.load(config[-2].strip())
    biases = torchfile.load(config[-1].strip())

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
            layer = Dropout(desc[1])
        else:
            print(desc[0] + ' layer not implemented!')
        model.addLayer(layer)

    return model


def main(args):
    model = getModel(args.config)
    input = torch.Tensor(torchfile.load(args.i))
    batch_size = input.size(0)
    input = input.view(batch_size, -1)
    gradients = torch.Tensor(torchfile.load(args.go))

    output = model(input)
    model.clearGradParam()
    model.backward(input, gradients)
    gradients = model.getGradients()

    torch.save(output.numpy(), args.o)
    torch.save(gradients['gradW'], args.ow)
    torch.save(gradients['gradB'], args.ob)
    torch.save(gradients['gradInput'], args.ig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', help='Path to the model configuration file.', required=True)
    parser.add_argument('-i', help='Inputs to the model.', required=True)
    parser.add_argument('-go', help='Gradients of the outputs.', required=True)
    parser.add_argument('-o', help='Path to save the outputs.', required=True)
    parser.add_argument('-ow', help='Gradients w.r.t. model weights.', required=True)
    parser.add_argument('-ob', help='Gradients w.r.t. model biases.', required=True)
    parser.add_argument('-ig', help='Gradients w.r.t. inputs.', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
