local TemporalConvolution, parent = torch.class('nn.TemporalConvolution', 'nn.Module')

function TemporalConvolution:__init(inputFrameSize, outputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.bias = torch.Tensor(outputFrameSize)
   self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.gradBias = torch.Tensor(outputFrameSize)
   
   self:reset()
end

function TemporalConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end
   self.weight:apply(function()
                        return random.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                      return random.uniform(-stdv, stdv)
                   end)   
end

function TemporalConvolution:forward(input)
   return input.nn.TemporalConvolution_forward(self, input)
end

function TemporalConvolution:backward(input, gradOutput)
   if self.gradInput then
      return input.nn.TemporalConvolution_backward(self, input, gradOutput)
   end
end

function TemporalConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.nn.TemporalConvolution_accGradParameters(self, input, gradOutput, scale)
end

-- function TemporalConvolution:accUpdateGradParameters(input, gradOutput, lr)
--    print('using slow version')
--    self:zeroGradParameters()
--    self:accGradParameters(input, gradOutput)
--    self:updateParameters(lr)
-- end
