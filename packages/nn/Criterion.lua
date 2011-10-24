local Criterion = torch.class('nn.Criterion')

function Criterion:__init()
   self.gradInput = torch.Tensor()
   self.output = 0
end

function Criterion:forward(input, target)
end

function Criterion:backward(input, target)
end

function Criterion:write(file)
   file:writeObject(self.gradInput)
   file:writeDouble(self.output) 
end

function Criterion:read(file)
   self.gradInput = file:readObject()
   self.output = file:readDouble()
end

function Criterion:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function Criterion:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
      end
   end
   return self
end

function Criterion:float()
   return self:type('torch.FloatTensor')
end

function Criterion:double()
   return self:type('torch.DoubleTensor')
end

function Criterion:cuda()
   return self:type('torch.CudaTensor')
end
