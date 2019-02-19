class Model():
    def __init__(self, layers=[]):
        self.layers = []

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

    def backward(self, input, gradOutput):
        for i in range(len(self.layers)-1,1,-1):
            gradOutput = self.layers[i].backward(self.layers[i-1].output, gradOutput)
        self.layers[0].backward(input, gradOutput)

    def clearGradParam(self):
        for layer in self.layers:
            layer.clearGradParam()

    def addLayer(self, layer):
        self.layers.append(layer)
