import argparse
import torchfile
import torch
from src.Criterion import Criterion
torch.set_default_tensor_type(torch.DoubleTensor)


def main(args):
    criterion = Criterion()
    input = torch.Tensor(torchfile.load(args.i))
    target = torch.Tensor(torchfile.load(args.t)) - 1  # because python uses 0-indexing

    loss = criterion(input, target)
    print('Average cross-entropy loss = {}'.format(loss))

    gradInput = criterion.backward(input, target)
    torch.save(gradInput.numpy(), args.ig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='Inputs to the criterion.', required=True)
    parser.add_argument('-t', help='Target labels.', required=True)
    parser.add_argument('-ig', help='Gradients w.r.t. inputs.', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
