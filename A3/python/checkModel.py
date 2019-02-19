import argparse
import os
import torchfile
import torch
from Model import Model
from Linear import Linear
from ReLU import ReLU


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
            layer = Linear(int(desc[1]), int(desc[2]))
            layer.W = torch.Tensor(weights[il])
            layer.B = torch.Tensor(biases[il])
            il += 1
        elif desc[0] == 'relu':
            layer = ReLU()
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

    torch.save(output, args.o)
    torch.save(gradients['gradW'], args.ow)
    torch.save(gradients['gradB'], args.ob)
    torch.save(gradients['gradInput'], args.ig)


def parse_args():
    parser = argparse.ArgumentParser()
    base_dir = '../CS 763 Deep Learning HW'
    output_dir = '../outputs'
    parser.add_argument('-config', help='Path to the model configuration file.', default=os.path.join(base_dir, 'modelConfig_2.txt'))
    parser.add_argument('-i', help='Inputs to the model.', default=os.path.join(base_dir, 'input_sample_2.bin'))
    parser.add_argument('-go', help='Gradients of the outputs.', default=os.path.join(base_dir, 'gradOutput_sample_2.bin'))
    parser.add_argument('-o', help='Path to save the outputs.', default=os.path.join(output_dir, 'output_2.bin'))
    parser.add_argument('-ow', help='Gradients w.r.t. model weights.', default=os.path.join(output_dir, 'gradw_2.bin'))
    parser.add_argument('-ob', help='Gradients w.r.t. model biases.', default=os.path.join(output_dir, 'gradb_2.bin'))
    parser.add_argument('-ig', help='Gradients w.r.t. inputs.', default=os.path.join(output_dir, 'gradi_2.bin'))
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
