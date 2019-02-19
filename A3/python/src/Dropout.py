import torch
from src.Layer import Layer


class Dropout(Layer):
    def __init__(self, prob_keep):
        super(Dropout, self).__init__()
        self.prob_keep = prob_keep

    def __repr__(self):
        return 'Dropout-{}'.format(self.prob_keep)

    def forward(self, input):
        prob_tensor = torch.Tensor(1,input.size(1))
        prob_tensor[:] = self.prob_keep
        self.mask = torch.bernoulli(prob_tensor).expand_as(input)  # self.mask is of size : layer_dim x layer_dim

        self.output = input*self.mask
        return self.output

    def backward(self, input, gradOutput):
        self.gradInput = gradOutput.mm(self.mask)
        return self.gradInput
