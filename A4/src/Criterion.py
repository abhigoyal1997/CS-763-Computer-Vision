''' For the special case when it is binary classification (ie, only one logit)
    Assuming input.shape == (bs,seq_length,1) and target.shape == (bs,)
'''
import torch

class Criterion():
    def __call__(self, input, lengths, target):
        return self.forward(input, lengths, target)

    def forward(self, input, lengths, target):
        # print(input.size(0))
        logits = torch.Tensor([input[i,lengths[i]-1,0] for i in range(input.size(0))])
        p_pred = logits.sigmoid()
        p_true = target.squeeze().double()
        return -(p_true.mul(p_pred.log()) + (1-p_true).mul((1-p_pred).log())).mean()

    def backward(self, input, lengths, target):
        onoff = torch.zeros_like(input)
        for i in range(input.size(0)):
            onoff[i,lengths[i]-1,0] = 1.0
        p_pred = input.double().sigmoid()
        p_true = target.double().expand_as(p_pred.squeeze().t()).t()
        p_true = p_true.unsqueeze(dim=2)
        # print(p_true.shape)
        # print(p_pred.shape)
        gradInput = p_pred - p_true
        batch_size = target.size(0)
        gradInput /= batch_size
        gradInput = gradInput.mul(onoff)
        return gradInput
