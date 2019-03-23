class Optimizer:
    def __init__(self, model, lr, lr_decay=1, momentum=0):
        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum

    def updateStep(self):
        for layer in self.model.layers:
            layer.gradientStep(self.lr, self.momentum)

    def updateLR(self):
        self.lr *= self.lr_decay