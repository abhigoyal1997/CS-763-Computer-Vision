import torch
from src.Layer import Layer


def convolution2D(input, filter):
    input_size = input.size(1)
    filter_size = filter.size(1)
    out_size = input_size - filter_size + 1

    xf = torch.arange(0,out_size).view(out_size,1)
    xf = torch.cat([xf+i for i in range(filter_size)], dim=1)
    xf = torch.cat([xf+i*input_size for i in range(filter_size)], dim=1)
    xf = torch.cat([xf+i*input_size for i in range(out_size)], dim=0)

    xx = input.view(-1,input_size*input_size)[:,xf]
    ff = filter.view(-1,filter_size*filter_size, 1)
    output = torch.matmul(xx, ff).view(-1,out_size,out_size)
    return output


class SimpleConvolution2D(Layer):
    def __init__(self, in_features, out_features, kernel_size):
        super(SimpleConvolution2D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.W = torch.randn(out_features, in_features, kernel_size, kernel_size)*0.1
        self.B = torch.zeros(out_features)

        self.gradW = torch.zeros(self.W.shape)
        self.gradB = torch.zeros(self.B.shape)

    def __repr__(self):
        return 'SimpleConvolution2D-{}'.format(tuple(self.W.t().shape))

    def forward(self, input):
        self.output = convolution2D(input, self.W.expand(input.size(0), -1, -1, -1, -1))
        self.output += self.B
        return self.output

    def backward(self, input, gradOutput):
        self.gradW = convolution2D(input, gradOutput).sum(0)
        pad = self.kernel_size - 1
        grad_padded = torch.zeros(gradOutput.size(0), 2*pad+gradOutput.size(1), 2*pad+gradOutput.size(1))
        grad_padded[:, pad:pad+gradOutput.size(1), pad:pad+gradOutput.size(2)] = gradOutput
        self.gradInput = convolution2D(grad_padded, self.W.expand(input.size(0), -1, -1))
        self.gradB = gradOutput.sum()
        return self.gradInput

    def clearGradParam(self):
        super(SimpleConvolution2D, self).clearGradParam()
        self.gradW[:] = 0
        self.gradB.fill_(0)
