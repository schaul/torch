local Max, parent = torch.class('nn.Max', 'nn.Module')

function Max:__init(dimension)
   parent.__init(self)
   dimension = dimension or 1
   self.dimension = dimension
   self.indices = torch.Tensor()
end

function Max:forward(input)
   return input.nn.Max_forward(self, input)
end

function Max:backward(input, gradOutput)
   return input.nn.Max_backward(self, input, gradOutput)
end
