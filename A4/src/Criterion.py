# For the special case when it is binary classification (ie, only one logit)
class Criterion():
    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        scaled_logits = input - input.max(dim=1, keepdim=True)[0]
        return (-scaled_logits[range(target.size(0)),target.squeeze().long()] + scaled_logits.exp().sum(dim=1).log()).mean()

    def backward(self, input, target):
        scaled_logits = input - input.max(dim=1, keepdim=True)[0]
        gradInput = (scaled_logits).softmax(dim=1)
        batch_size = target.size(0)
        gradInput[range(batch_size),target.squeeze().long()] -= 1
        gradInput /= batch_size
        return gradInput
