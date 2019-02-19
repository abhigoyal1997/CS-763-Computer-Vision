import torch
from Layer import Layer


class Linear(Layer):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.W = torch.rand(out_features, in_features)
        self.B = torch.rand(out_features, 1)

        self.gradW = torch.zeros(self.W.shape)
        self.gradB = torch.zeros(self.B.shape)

    def forward(self, input):
        self.output = input.mm(self.W.t())
        return self.output

    def backward(self, input, gradOutput):
        self.gradInput = gradOutput.mm(self.W)
        self.gradW = gradOutput.t().mm(input)
        self.gradB = gradOutput.sum(dim=1)
        return self.gradInput

    def clearGradParam(self):
        self.gradW[:] = 0
        self.gradB[:] = 0
