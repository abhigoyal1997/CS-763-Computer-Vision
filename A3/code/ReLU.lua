local class = require "class"

local ReLU = class('ReLU')

function ReLU:__init()
end

function ReLU:forward(input)
    self.output = torch.cmax(imput, 0)
    return self.output
end

function ReLU:backward(input, gradOutput)
    self.gradInput = torch.cmax(gradOutput, 0)
    return self.gradInput
end

function ReLU:clearGradParam()
end
