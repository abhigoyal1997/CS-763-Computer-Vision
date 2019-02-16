local class = require "class"

local Linear = class('Linear')

function Linear:__init(in_features, out_features)
    self.W = torch.Tensor(out_features, in_features)
    self.B = torch.Tensor(out_features, 1)

    self.gradW = torch.Tensor(self.W:size())
    self.gradB = torch.Tensor(self.B:size())
end

function Linear:forward(input)
    self.output = input*self.W:t()
    self.output = self.output + self.B:t():expandAs(self.output);
    return self.output
end

function Linear:backward(input, gradOutput)
    self.gradInput = gradOutput*self.W
    self.gradW = input*gradOutput:t()
    self.gradB = gradOutput:sum(2)

    return self.gradInput
end

function Linear:clearGradParam()
    self.gradW:fill(0)
    self.gradB:fill(0)
end
