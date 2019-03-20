import torch
from src.Layer import Layer


class SimpleMaxPool2D(Layer):
    def __init__(self, kernel_size):
        super(SimpleMaxPool2D, self).__init__()
        self.kernel_size = kernel_size

    def __repr__(self):
        return 'SimpleMaxPool2D-{}'.format(self.kernel_size)

    def forward(self, input):
        input_size = input.size(1)
        out_size = int(input_size/self.kernel_size)

        xx = input
        xx = torch.cat([
            torch.cat([
                xx[:,i+j,:].view(xx.size(0),-1,self.kernel_size) for j in range(self.kernel_size)
            ], dim=2) for i in range(0,input_size,self.kernel_size)
        ], dim=1)

        self.output, self.output_idx = xx.max(dim=2)
        self.output = self.output.view(-1, out_size, out_size)
        return self.output

    def backward(self, input, gradOutput):
        out_size = gradOutput.size(1)
        grad = torch.zeros(input.size(0), out_size**2, self.kernel_size**2)
        grad = grad.scatter(2, self.output_idx.unsqueeze(2), 1)
        grad = grad*gradOutput.view(-1,53*53).unsqueeze(2)
        self.gradInput = grad = torch.cat([
            torch.cat([
                grad[:,i+j,:].view(grad.size(0),-1,self.kernel_size) for j in range(out_size)
            ], dim=2) for i in range(0,grad.size(1),out_size)
        ], dim=1)

        return self.gradInput
