local Binary, parent = torch.class('bnn.Binary','nn.Module')

function Binary:updateOutput(input)
    if self.train then
        self.output:resizeAs(input):copy(input)
        self.output:sign()
        return self.output
    else
        return input:sign()
    end
end

function Binary:updateGradInput(input,gradOutput)
    self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    self.gradInput[input:ge(-1)] = 0
    self.gradInput[input:le(-1)] = 0
    return self.gradInput
end