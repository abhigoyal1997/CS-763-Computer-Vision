class Criterion():
    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        return (-input[range(10),target.long()] + (-input).exp().sum(dim=1).log()).sum()

    def backward(self, input, target):
        gradInput = input.softmax(dim=1)
        gradInput[target.long()] -= 1
        return gradInput
