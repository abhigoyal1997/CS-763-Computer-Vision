import torch
import math
from src.Layer import Layer


class RNN(Layer):
    def __init__(self, in_features, hidden_features, out_features, T):
        super(RNN, self).__init__()
        self.Whh = torch.randn(hidden_features, hidden_features)*math.sqrt(2.0/hidden_features)
        self.Bhh = torch.zeros(hidden_features, 1)

        self.Wxh = torch.randn(hidden_features, in_features)*math.sqrt(2.0/in_features)
        self.Bxh = torch.zeros(hidden_features, 1)

        self.Why = torch.randn(out_features, hidden_features)*math.sqrt(2.0/hidden_features)
        self.Bhy = torch.zeros(out_features, 1)        

        self.gradWhh = torch.zeros(self.Whh.shape)
        self.gradBhh = torch.zeros(self.Bhh.shape)

        self.gradWxh = torch.zeros(self.Wxh.shape)
        self.gradBxh = torch.zeros(self.Bxh.shape)

        self.gradWhy = torch.zeros(self.Why.shape)
        self.gradBhy = torch.zeros(self.Bhy.shape)

    def __repr__(self):
        return 'RNN-{}'.format(tuple(self.Wxh.t().shape))

    def forward(self, input):
        ## Assuming input is batch_sizeXseq.lengthXin_features
        hidden_vector = torch.zeros_like(input[:][0].mm(self.Wxh.t()))
        for i in range(T):
            temp1 = hidden_vector.mm(self.Whh.t())
            temp1 = temp1 + self.Bhh.t().expand_as(temp1)

            temp2 = input[:][i].mm(self.Wxh.t())
            temp2 = temp2 + self.Bxh.t().expand_as(temp2)  

            hidden_vector = torch.tanh(temp1 + temp2)

        self.output = hidden_vector.mm(Why.t())
        self.output = self.output + Bhy.t().expand_as(self.output)
        return self.output

    def backward(self, input, gradOutput):
        # TODO
        pass

    def clearGradParam(self):
        super(RNN, self).clearGradParam()
        self.gradWhh[:] = 0
        self.gradBhh[:] = 0
        self.gradWxh[:] = 0
        self.gradBxh[:] = 0
        self.gradWhy[:] = 0
        self.gradBhy[:] = 0
