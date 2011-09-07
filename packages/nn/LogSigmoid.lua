local LogSigmoid, parent = torch.class('nn.LogSigmoid', 'nn.Module')

function LogSigmoid:__init()
   parent.__init(self)
   self.buffer = torch.Tensor()
end

function LogSigmoid:forward(input)
   return input.nn.LogSigmoid_forward(self, input)
end

function LogSigmoid:backward(input, gradOutput)
   return input.nn.LogSigmoid_backward(self, input, gradOutput)
end
