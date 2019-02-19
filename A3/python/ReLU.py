from Layer import Layer


class ReLU(Layer):
    def forward(self, input):
        self.output = input.clone()
        self.output[input<0] = 0
        return self.output

    def backward(self, input, gradOutput):
        self.gradInput = gradOutput
        self.gradInput[input<0] = 0
        return self.gradInput
