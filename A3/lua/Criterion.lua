local Criterion = torch.class('Criterion')

function Criterion:forward(input, target)
    return (-input:gather(2, target:long():resize(target:size(1),1)) + torch.log(torch.sum(torch.exp(-input),2))):sum(1)
end

function Criterion:backward(input, gradOutput)
    -- todo: implement this
end

return Criterion
