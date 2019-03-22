import torch
import math
from src.Layer import Layer

class RNN(Layer):
    def __init__(self, in_features, hidden_features, out_features):
        super(RNN, self).__init__()
        self.Whh = torch.randn(hidden_features, hidden_features)*math.sqrt(2.0/hidden_features)
        self.Bhh = torch.zeros(hidden_features, 1)

        self.Wxh = torch.randn(hidden_features, in_features)*math.sqrt(2.0/in_features)
        # self.Bxh = torch.zeros(hidden_features, 1) #@Argument1

        self.Why = torch.randn(out_features, hidden_features)*math.sqrt(2.0/hidden_features)
        self.Bhy = torch.zeros(out_features, 1)        

        self.gradWhh = torch.zeros(self.Whh.shape)
        self.gradBhh = torch.zeros(self.Bhh.shape)

        self.gradWxh = torch.zeros(self.Wxh.shape)
        # self.gradBxh = torch.zeros(self.Bxh.shape) #@Argument1

        self.gradWhy = torch.zeros(self.Why.shape)
        self.gradBhy = torch.zeros(self.Bhy.shape)

    def __repr__(self):
        return 'RNN-{}'.format(tuple(self.Wxh.t().shape))

    def forward(self, input):
        ## Assuming input is batch_size X seq_length X in_features
        _, time_steps, _ = list(input.shape)
        hidden_vector = torch.zeros_like(input[:,0,:].mm(self.Wxh.t())) # batch_size x hidden_features
        self.hidden_states = [hidden_vector] # also storing the initial zero-state
        for t in range(time_steps):
            temp_1 = hidden_vector.mm(self.Whh.t())
            temp_1 = temp_1 + self.Bhh.t().expand_as(temp_1)

            temp_2 = input[:,t,:].mm(self.Wxh.t())
            # temp_2 = temp_2 + self.Bxh.t().expand_as(temp_2) #@Argument1

            hidden_vector = torch.tanh(temp_1 + temp_2)
            self.hidden_states.append(hidden_vector)

        temp_output = hidden_vector.mm(self.Why.t())
        self.output = temp_output + self.Bhy.t().expand_as(temp_output)
        return self.output

    def backward(self, input, gradOutput):
        _, time_steps, _ = list(input.shape)
        # Pass the gradients through the output layer
        # gradOutput - batch_size x out_features
        self.gradWhy = gradOutput.t().mm(self.hidden_states[-1])
        self.gradBhy = gradOutput.sum(dim=0, keepdim=True).t()
        gradOutput = gradOutput.mm(self.Why)
        # Loop over time and pass the gradients through hidden and input layers
        # gradOutput - batch_size x hidden_features
        for t in range(time_steps,0,-1):
            gradOutput = gradOutput.mul(1 - self.hidden_states[t]**2)
            self.gradWhh += gradOutput.t().mm(self.hidden_states[t-1])
            self.gradBhh += gradOutput.sum(dim=0, keepdim=True).t()
            self.gradWxh += gradOutput.t().mm(input[:,t-1,:])
            gradOutput = gradOutput.mm(self.Whh)
        return gradOutput

    def clearGradParam(self):
        super(RNN, self).clearGradParam()
        self.gradWhh[:] = 0
        self.gradBhh[:] = 0
        self.gradWxh[:] = 0
        self.gradBxh[:] = 0
        self.gradWhy[:] = 0
        self.gradBhy[:] = 0

'''
Argument1 : Two bias terms for Bhh and Bxh not needed, since temp_1 and temp_2 are added before tanh()
            So essentially, we only need one bias term for these two. We keep Bhh.
'''