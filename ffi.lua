local ffi = require 'ffi'

local libpath = package.searchpath('libbnn',package.cpath)
if not libpath then return end

require 'cunn'

ffi.cdef[[
        void BinarySpatialConvolution_updateOutput(
                THCState *state,
                THCudaTensor *input,
                THCudaTensor *output,
                THCudaIntTensor *weight,
                THCudaTensor *columns,
                THCudaTensor *alphas,
                THCudaIntTensor *columns_binary,
                int kW, int kH,
                int dW, int dH,
                int padW, int padH);

        void encode_rows(
                THCState *state, 
                THCudaTensor* input, 
                THCudaIntTensor* output);
	
        void encode_cols(
                THCState *state, 
                THCudaTensor* input, 
                THCudaIntTensor* output);

        void decode(
                THCState *state, 
                THCudaIntTensor* input, 
                THCudaTensor* output);
]]

return ffi.load(libpath)