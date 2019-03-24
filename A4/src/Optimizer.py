class Optimizer:
    def __init__(self, model, lr, lr_decay=1, alpha=0.9, zeta=0.99, eps=1e-7):
        self.model = model
        self.lr = lr
        self.lr_decay = lr_decay
        self.alpha = alpha
        self.zeta = zeta
        self.eps = eps

    def updateStep(self):
        for layer in self.model.layers:
            layer.gradientStep(self.lr, self.alpha, self.zeta, self.eps)

    def updateLR(self):
        self.lr *= self.lr_decay