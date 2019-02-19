import torch


class Linear():
    def __init__(self, in_features, out_features):
        self.W = torch.rand(out_features, in_features)
        self.B = torch.rand(out_features, 1)

        self.gradW = torch.zeros(self.W.shape)
        self.gradB = torch.zeros(self.B.shape)

        self.output = torch.Tensor()
        self.gradInput = torch.Tensor()

    def forward(self, input):
        self.output = input*self.W.T
        return self.output

    def backward(self, input, gradOutput):
        self.gradInput = gradOutput*self.W
        self.gradW = gradOutput.T*input
        self.gradB = gradOutput.sum(dim=1)
        return self.gradInput
