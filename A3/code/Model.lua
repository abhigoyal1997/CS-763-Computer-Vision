local class = require "class"

local Model = class('Model')

function Model:__init()
    self.layers = {}
end

function Model:forward(input)
    output = input:clone()
    for _,v in ipairs(self.layers) do
        output = v.forward(output)
    end
    return output
end

function Model:backward(input, gradOutput)
    for i=#self.layers,2,-1 do
        gradOutput = self.layers[i].backward(self.layers[i-1].output, gradOutput)
    end
    self.layers[1].backward(input, gradOutput)
end

function Model:dispGradParam()

end

function Model:clearGradParam()
    for i=1,#self.layers do
        self.layers[i].clearGradParam()
    end
end

function Model:addLayer(layer)
    table.insert(self.layers, layer)
end
