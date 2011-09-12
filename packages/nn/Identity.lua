local Identity, parent = torch.class('nn.Identity', 'nn.Module')

function Identity:forward(input)
   self.output = input
   return self.output
end


function Identity:backward(input, gradOutput)
   self.gradInput = gradOutput
   return self.gradInput
end
