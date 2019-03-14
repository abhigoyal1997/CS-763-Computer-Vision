class Layer():
    def __init__(self):
        self.gradInput = None

    def __call__(self, input):
        return self.forward(input)

    def __repr__(self):
        return 'Generic Layer'

    def forward(self, input):
        pass

    def backward(self, input, gradOutput):
        pass

    def clearGradParam(self):
        if self.gradInput is not None:
            self.gradInput[:] = 0
