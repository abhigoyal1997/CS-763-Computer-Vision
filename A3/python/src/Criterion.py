class Criterion():
    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        return (-input[range(target.size(0)),target.squeeze().long()] + input.exp().sum(dim=1).log()).mean()

    def backward(self, input, target):
        gradInput = input.softmax(dim=1)
        batch_size = target.size(0)
        gradInput[range(batch_size),target.squeeze().long()] -= 1
        gradInput /= batch_size
        return gradInput
