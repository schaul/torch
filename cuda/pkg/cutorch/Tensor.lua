function torch.CudaTensor.apply(self, func)
   local x = torch.FloatTensor(self:size()):copy(self)
   x:apply(func)
   self:copy(x)
end

local function Tensor__type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end

local function Tensor__cuda(self,type)
   return self:type('torch.CudaTensor')
end
local function Tensor__double(self,type)
   return self:type('torch.DoubleTensor')
end
local function Tensor__float(self,type)
   return self:type('torch.FloatTensor')
end

rawset(torch.getmetatable('torch.DoubleTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.FloatTensor'), 'cuda', Tensor__cuda)
rawset(torch.getmetatable('torch.CudaTensor'), 'cuda', Tensor__cuda)

rawset(torch.getmetatable('torch.CudaTensor'), 'type', Tensor__type)
rawset(torch.getmetatable('torch.CudaTensor'), 'double', Tensor__double)
rawset(torch.getmetatable('torch.CudaTensor'), 'float', Tensor__float)
