from src.Layer import Layer


class Flat(Layer):
    def __repr__(self):
        return 'Flat'

    def forward(self, input):
        self.output = input.view(input.size(0),-1)
        return self.output

    def backward(self, input, gradOutput):
        self.gradInput = gradOutput.view_as(input)
        return self.gradInput
