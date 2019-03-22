# For the special case when it is binary classification (ie, only one logit)
# Assuming input.shape == (bs,1) and target.shape == (bs,)
class Criterion():
    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        input_dimmed = input.squeeze().double() # shape == (bs,)
        p_pred = input_dimmed.sigmoid()
        p_true = target.squeeze().double()
        return -(p_true.mul(p_pred.log()) + (1-p_true).mul((1-p_pred).log())).mean()

    def backward(self, input, target):
        input_dimmed = input.squeeze().double() # shape == (bs,)
        p_pred = input_dimmed.sigmoid()
        p_true = target.squeeze().double()
        gradInput = (p_pred - p_true)
        batch_size = target.size(0)
        gradInput /= batch_size
        return gradInput
