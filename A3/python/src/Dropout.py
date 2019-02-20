import torch
from src.Layer import Layer


class Dropout(Layer):
    def __init__(self, prob_keep, isTrain):
        super(Dropout, self).__init__()
        self.prob_keep = prob_keep
        self.isTrain = isTrain

    def __repr__(self):
        return 'Dropout-{}'.format(self.prob_keep)

    def forward(self, input):
        if self.isTrain:
            prob_tensor = torch.Tensor(1,input.size(1))
            prob_tensor[:] = self.prob_keep
            self.mask = torch.bernoulli(prob_tensor).expand_as(input)
            self.output = input*self.mask
        else:
            self.output = input.clone()
        return self.output

    def backward(self, input, gradOutput):
        if self.isTrain:
            self.gradInput = gradOutput*self.mask
        else:
            self.gradInput = gradOutput.clone()
        return self.gradInput
