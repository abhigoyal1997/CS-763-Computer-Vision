local class = require "class"

local Criterion = class('Criterion')

function Criterion:forward(input, target)
    return (-inp:gather(2, target:long():resize(#target,1)) + torch.log(torch.sum(torch.exp(-inp),2))):sum(1)
end

function Criterion:backward(input, gradOutput)
    -- todo: implement this
end
