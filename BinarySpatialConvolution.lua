local C = bnn.C
local BinarySpatialConvolution, parent = torch.class('bnn.SpatialConvolution','nn.Module')

function BinarySpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH

   self.dW = dW
   self.dH = dH
   self.padW = padW or 0
   self.padH = padH or self.padW
   
   self.alphas = torch.Tensor(nOutputPlane)

   self.finput = self.finput or self.alphas.new() 
end

function BinarySpatialConvolution:updateOutput(input)
  -- Binary columnation buffer
  self.columns_binary = self.columns_binary or torch.CudaIntTensor()
  
  C.BinarySpatialConvolution_updateOutput(
        cutorch.getState(),
        input:cdata(),
        self.output:cdata(),
        self.weight:cdata(),
        self.finput:cdata(),
		    self.alphas:cdata(),
		    self.columns_binary:cdata(),
        self.kW, self.kH,
        self.dW, self.dH,
        self.padW, self.padH
   )

  return self.output
end

function BinarySpatialConvolution:noBias()
    return self
end

-- Helper function fo convert real weights to their binary form
-- Expects a convolutional layer that is ready to be binarised
function BinarySpatialConvolution:binarise(convLayer)
  if torch.typename(convLayer) ~= 'nn.SpatialConvolution' and torch.typename(convLayer) ~= 'cudnn.SpatialConvolution' then
    print("Only convolutional layers can be used as input")
    return;
  end

  local wSize = convLayer.weight:size()
  self.alphas = convLayer.weight:norm(1,4):sum(3):sum(2):div(convLayer.weight[1]:nElement())
  convLayer.weight = torch.CudaTensor(convLayer.weight:storage(),convLayer.weight:storageOffset(),wSize[1],-1,wSize[2]*wSize[3]*wSize[4],-1)
  convLayer.weight:sign()
  
  self.weight = torch.CudaIntTensor(self.nOutputPlane, self.nInputPlane*self.kH*self.kW/32)
  C.encode_rows(cutorch.getState(),convLayer.weight:cdata(),self.weight:cdata())
end

function BinarySpatialConvolution:type(type,tensorCache)
   self.finput = self.finput and torch.Tensor()
   return parent.type(self,type,tensorCache)
end

function BinarySpatialConvolution:__tostring__()
   local s = string.format('%s(%d -> %d, %dx%d', torch.type(self),
         self.nInputPlane, self.nOutputPlane, self.kW, self.kH)
   if self.dW ~= 1 or self.dH ~= 1 or self.padW ~= 0 or self.padH ~= 0 then
     s = s .. string.format(', %d,%d', self.dW, self.dH)
   end
   if (self.padW or self.padH) and (self.padW ~= 0 or self.padH ~= 0) then
     s = s .. ', ' .. self.padW .. ',' .. self.padH
   end
  
   return s .. ') without bias'
end

function BinarySpatialConvolution:clearState()
   nn.utils.clear(self, 'finput', 'fgradInput', 'columns_binary ', '_input', '_gradOutput')
   return parent.clearState(self)
end


