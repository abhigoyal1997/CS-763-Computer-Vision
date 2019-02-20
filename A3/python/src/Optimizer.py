import torch
from src.Linear import Linear


class Optimizer:
    def __init__(self, model, lr, lr_decay=1, momentum=0):
        self.model = model
        self.num_layers = len(model.layers)
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.step = [None]*self.num_layers
        for layer_index in range(self.num_layers):
            layer = self.model.layers[layer_index]
            if isinstance(layer, Linear):
                self.step[layer_index] = [torch.zeros(layer.W.shape), torch.zeros(layer.B.shape)]

    def updateStep(self):
        for layer_index in range(self.num_layers):
            layer = self.model.layers[layer_index]
            if isinstance(layer, Linear):
                step_W_prev, step_B_prev = self.step[layer_index]
                step_W = self.momentum*step_W_prev + self.lr*layer.gradW
                step_B = self.momentum*step_B_prev + self.lr*layer.gradB
                layer.W -= step_W
                layer.B -= step_B
                self.step[layer_index] = [step_W, step_B]

    def updateLR(self):
        self.lr *= self.lr_decay
