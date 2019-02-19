import torch
from Layer import Layer


class Dropout(Layer):
    def __init__(self, prob_keep):
        super(Dropout, self).__init__()
        self.prob_keep = prob_keep
    
    def forward(self, input):
        layer_dim = input.size()[1] # assuming input is batch_size x layer_dim
        prob_tensor = torch.tensor([prob_keep]*layer_dim)
        mask_sampler = torch.distributions.bernoulli.Bernoulli(prob_tensor)
        mask_vector = mask_sampler.sample()
        # self.mask is of size : layer_dim x layer_dim
        self.mask = mask_vector.repeat(layer_dim, 1).t() # Each column is the same
        self.output = input.clone().mm(self.mask)
        return self.output

    def backward(self, input, gradOutput):
        self.gradInput = gradOutput.mm(self.mask)
        return self.gradInput