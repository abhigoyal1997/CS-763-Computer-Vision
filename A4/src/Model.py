class Model():
    def __init__(self, layers=[], H=None, V=None, D=None):
        self.layers = layers
        self.nLayers = len(layers)
        self.H = H
        self.V = V
        self.D = D

    def __call__(self, input):
        return self.forward(input)

    def __repr__(self):
        return 'Model-{}'.format(self.layers)

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer(output)
        return output

    def backward(self, input, gradOutput):
        for i in range(len(self.layers)-1,0,-1):
            gradOutput = self.layers[i].backward(self.layers[i-1].output, gradOutput)
        self.layers[0].backward(input, gradOutput)

    def clearGradParam(self):
        for layer in self.layers:
            layer.clearGradParam()

    def addLayer(self, layer):
        self.layers.append(layer)
        self.nLayers += 1

    def getGradients(self):
        gradients = {}
        for x in ['gradWhh','gradWxh','gradWhy','gradBhh','gradBhy', 'gradInput']: #removed 'gradBxh'
            gradients[x] = []
            for layer in self.layers:
                if hasattr(layer, x):
                    gradients[x].append(eval('layer.'+x).numpy())
        return gradients

    def getParams(self):
        params = {}
        for x in ['Whh','Wxh','Why','Bhh','Bhy']: #removed 'Bxh'
            params[x] = []
            for layer in self.layers:
                if hasattr(layer, x):
                    params[x].append(eval('layer.'+x).numpy())
        return params
