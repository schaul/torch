local SpatialMaxPooling, parent = torch.class('nn.SpatialMaxPooling', 'nn.Module')

function SpatialMaxPooling:__init(kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.indices = torch.Tensor()
end

function SpatialMaxPooling:forward(input)
   input.nn.SpatialMaxPooling_forward(self, input)
   return self.output
end

function SpatialMaxPooling:backward(input, gradOutput)
   input.nn.SpatialMaxPooling_backward(self, input, gradOutput)
   return self.gradInput
end

function SpatialMaxPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end

function SpatialMaxPooling:write(file)
   parent.write(self, file)
   file:writeInt(self.kW)
   file:writeInt(self.kH)
   file:writeInt(self.dW)
   file:writeInt(self.dH)
   file:writeObject(self.indices)
end

function SpatialMaxPooling:read(file)
   parent.read(self, file)
   self.kW = file:readInt()
   self.kH = file:readInt()
   self.dW = file:readInt()
   self.dH = file:readInt()
   self.indices = file:readObject()
end
