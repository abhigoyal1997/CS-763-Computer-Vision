class Layer():
    def __init__(self):
        pass

    def forward(self, input):
        pass

    def backward(self, input, gradOutput):
        pass

    def __call__(self, input):
        return self.forward(input)
