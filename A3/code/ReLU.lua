local ReLU = torch.class('ReLU')

function ReLU:__init()
end

function ReLU:forward(input)
    self.output = torch.cmax(input, 0.0)
    return self.output
end

function ReLU:backward(input, gradOutput)
    self.gradInput = torch.cmul(torch.cmax(input, 0.0),gradOutput)
    return self.gradInput
end

function ReLU:clearGradParam()
    if self.gradInput ~= nil then
        self.gradInput:fill(0)
    end
end

return ReLU
