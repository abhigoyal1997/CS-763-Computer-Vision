import torch
import math
import numpy as np
from src.Layer import Layer

class RNN(Layer):
    def __init__(self, in_features, hidden_features, out_features, grad_thresh=1e1, back_thresh=1e-4):
        super(RNN, self).__init__()
        self.Whh = torch.randn(hidden_features, hidden_features)*math.sqrt(2.0/hidden_features)
        self.Bhh = torch.zeros(hidden_features, 1)

        self.Wxh = torch.randn(hidden_features, in_features)*math.sqrt(2.0/in_features)
        # self.Bxh = torch.zeros(hidden_features, 1) #@Reason1

        self.Why = torch.randn(out_features, hidden_features)*math.sqrt(2.0/hidden_features)
        self.Bhy = torch.zeros(out_features, 1)

        self.gradWhh = torch.zeros(self.Whh.shape)
        self.gradBhh = torch.zeros(self.Bhh.shape)

        self.gradWxh = torch.zeros(self.Wxh.shape)

        self.gradWhy = torch.zeros(self.Why.shape)
        self.gradBhy = torch.zeros(self.Bhy.shape)

        ## For momentum step storing 
        self.mWhh = torch.zeros(self.Whh.shape)
        self.mBhh = torch.zeros(self.Bhh.shape)

        self.mWxh = torch.zeros(self.Wxh.shape)

        self.mWhy = torch.zeros(self.Why.shape)
        self.mBhy = torch.zeros(self.Bhy.shape)

        self.cWhh = torch.zeros(self.Whh.shape)
        self.cBhh = torch.zeros(self.Bhh.shape)

        self.cWxh = torch.zeros(self.Wxh.shape)

        self.cWhy = torch.zeros(self.Why.shape)
        self.cBhy = torch.zeros(self.Bhy.shape)

        self.grad_thresh = grad_thresh
        self.back_thresh = back_thresh

    def __repr__(self):
        return 'RNN-{}'.format(tuple(self.Wxh.t().shape))

    def forward(self, input):
        ## Assuming input is batch_size X seq_length X in_features
        _, time_steps, _ = list(input.shape)
        hidden_vector = torch.zeros_like(input[:,0,:].mm(self.Wxh.t())) # batch_size x hidden_features
        self.hidden_states = [hidden_vector] # also storing the initial zero-state
        self.output = [] # will populate with the unrolled outputs of each cell
        for t in range(time_steps):
            temp = hidden_vector.mm(self.Whh.t())
            temp = temp + self.Bhh.t().expand_as(temp)
            temp = temp + input[:,t,:].mm(self.Wxh.t())

            hidden_vector = torch.tanh(temp)
            self.hidden_states.append(hidden_vector)

            temp_output = hidden_vector.mm(self.Why.t())
            output = temp_output + self.Bhy.t().expand_as(temp_output)
            self.output.append(output)
        self.output = torch.stack(self.output, dim=1)
        return self.output

    def backward(self, input, gradOutput):
        ## Assuming gradOutput is batch_size X seq_length X out_features, input as before
        _, time_steps, _ = list(input.shape)
        self.gradInput = torch.zeros_like(input) #@Reason2
        for t in range(time_steps-1,-1,-1):
            gradOutput_ = gradOutput[:,t,:] # batch_size x out_features
            if gradOutput_.norm() > self.back_thresh:
                self.gradWhy += gradOutput_.t().mm(self.hidden_states[t+1])
                self.gradBhy += gradOutput_.sum(dim=0, keepdim=True).t()
                gradOutput_ = gradOutput_.mm(self.Why) # batch_size x hidden_features
                ## Loop over time and pass the gradients through hidden and input layers
                for bptt in range(t+1,0,-1):
                    gradOutput_ = gradOutput_.mul(1 - self.hidden_states[bptt]**2)
                    self.gradWhh += gradOutput_.t().mm(self.hidden_states[bptt-1])
                    self.gradBhh += gradOutput_.sum(dim=0, keepdim=True).t()
                    self.gradWxh += gradOutput_.t().mm(input[:,bptt-1,:])
                    self.gradInput[:,bptt-1,:] += gradOutput_.mm(self.Wxh) # batch_size x in_features
                    gradOutput_ = gradOutput_.mm(self.Whh)
        self.clipGradients()
        return self.gradInput

    def clipGradients(self):
        normWhh = (self.gradWhh**2).sum().sqrt()
        if normWhh > self.grad_thresh:
            self.gradWhh *= (self.grad_thresh/normWhh)
        normBhh = (self.gradBhh**2).sum().sqrt()
        if normBhh > self.grad_thresh:
            self.gradBhh *= (self.grad_thresh/normBhh)

        normWxh = (self.gradWxh**2).sum().sqrt()
        if normWxh > self.grad_thresh:
            self.gradWxh *= (self.grad_thresh/normWxh)

        normWhy = (self.gradWhy**2).sum().sqrt()
        if normWhy > self.grad_thresh:
            self.gradWhy *= (self.grad_thresh/normWhy)
        normBhy = (self.gradBhy**2).sum().sqrt()
        if normBhy > self.grad_thresh:
            self.gradBhy *= (self.grad_thresh/normBhy)

    def gradientStep(self, lr, alpha, zeta, eps):
        self.mWhh = alpha*self.mWhh + (1-alpha)*self.gradWhh
        self.cWhh = zeta*self.cWhh + (1-zeta)*self.gradWhh**2
        self.stepWhh = -1*lr*self.mWhh/(np.sqrt(self.cWhh) + eps)

        self.mBhh = alpha*self.mBhh + (1-alpha)*self.gradBhh
        self.cBhh = zeta*self.cBhh + (1-zeta)*self.gradBhh**2
        self.stepBhh = -1*lr*self.mBhh/(np.sqrt(self.cBhh) + eps)

        self.mWxh = alpha*self.mWxh + (1-alpha)*self.gradWxh
        self.cWxh = zeta*self.cWxh + (1-zeta)*self.gradWxh**2
        self.stepWxh = -1*lr*self.mWxh/(np.sqrt(self.cWxh) + eps)

        self.mWhy = alpha*self.mWhy + (1-alpha)*self.gradWhy
        self.cWhy = zeta*self.cWhy + (1-zeta)*self.gradWhy**2
        self.stepWhy = -1*lr*self.mWhy/(np.sqrt(self.cWhy) + eps)

        self.mBhy = alpha*self.mBhy + (1-alpha)*self.gradBhy
        self.cBhy = zeta*self.cBhy + (1-zeta)*self.gradBhy**2
        self.stepBhy = -1*lr*self.mBhy/(np.sqrt(self.cBhy) + eps)

        self.Whh += self.stepWhh
        self.Bhh += self.stepBhh

        self.Wxh += self.stepWxh

        self.Why += self.stepWhy
        self.Bhy += self.stepBhy

    def clearGradParam(self):
        super(RNN, self).clearGradParam()
        self.gradWhh[:] = 0
        self.gradBhh[:] = 0
        self.gradWxh[:] = 0
        self.gradWhy[:] = 0
        self.gradBhy[:] = 0

'''
Reason1 : Two bias terms for Bhh and Bxh not needed, since terms are added before tanh().
            So essentially, we only need one bias term for these two. We keep Bhh.
Reason2 : Need to init here, because if we don't then it is None in first backward pass.
'''