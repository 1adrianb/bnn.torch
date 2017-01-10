// ------------------------------------------------------------------
// Spatial Binary Convolution
// Adrian Bulat,2017
// ------------------------------------------------------------------

#include "THC.h"
#include "common.h"
#include "THCNumerics.cuh"
#include "THCDeviceTensor.cuh"

#include <iostream>
#include <chrono>

#define BLOCK_SIZE 16

// Fallows the CUDA tutorial: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void gemm(float* A, float* B, float* C, int m, int n, int k, float* alphas) {
   // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

	int startLocation = BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol;
	
    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    float Cvalue = 0.0;
    
	int c = blockIdx.x*blockDim.x + threadIdx.x;  //row value using x-index of current thread
    int r = blockIdx.y*blockDim.y + threadIdx.y; //column value using y-index of current thread

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < 1+((n-1) / BLOCK_SIZE); ++i) {
        // Get sub-matrix Asub of A
        float* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        float* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix

        // Load zero if outside of the range
        if ((BLOCK_SIZE*i+col)<n && r<m)
            As[row][col] = Asub[row*n+col];
        else 
            As[row][col] = 0; 
        if ((BLOCK_SIZE*i+row)<n && c<k)
            Bs[row][col] = Bsub[row*k+col];
        else
            Bs[row][col] = 0;  
   
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
		#pragma unroll 4
        for (int j = 0; j < BLOCK_SIZE; ++j) { Cvalue += As[row][j] * Bs[j][col]; } 
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = Cvalue*alphas[(startLocation+row*k+col)/k];
}

// Based on the above one, the multiplication is replaced with binary operation popcount(xor)
// A is shape (m,n/32), B is shape (n/32,k) and C is shape (m,k)
__global__ void binary_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k, float *alphas) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

	int startLocation = BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol;
	
    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int Cvalue = 0;
	
	int c = blockIdx.x*blockDim.x + threadIdx.x;	//row value using x-index of current thread
    int r = blockIdx.y*blockDim.y + threadIdx.y; //column value using y-index of current thread
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < 1+((n-1) / BLOCK_SIZE); ++i) {
    
        // Get sub-matrix Asub of A
        unsigned int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        unsigned int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        // Load zero if outside of the range
        if ((BLOCK_SIZE*i+col)<n && r<m)
            As[row][col] = Asub[row*n+col];
        else 
            As[row][col] = 0; 
        if ((BLOCK_SIZE*i+row)<n && c<k)
            Bs[row][col] = Bsub[row*k+col];
        else
            Bs[row][col] = 0; 
		
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together (binary case)
        #pragma unroll 4
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += __popc(As[row][j]^Bs[j][col]);
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
	if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = -(2*(float)Cvalue-32*n)*alphas[(startLocation+row*k+col)/k];
}

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
template <typename Dtype>
__global__ void im2col_kernel(const int n, const Dtype* data_im,
                              const int height, const int width,
                              const int ksize_h, const int ksize_w,
                              const int pad_h, const int pad_w,
                              const int stride_h, const int stride_w,
                              const int dilation_h, const int dilation_w,
                              const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i * dilation_h;
        int w = w_in + j * dilation_w;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im[i * dilation_h * width + j * dilation_w] : ScalarConvert<int, Dtype>::to(-1);
        data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col(cudaStream_t stream, const Dtype* data_im, const int channels,
            const int height, const int width,
            const int ksize_h, const int ksize_w, const int pad_h,
            const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1))
                   / stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1))
                  / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // Launch
  im2col_kernel <<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>> (
      num_kernels, data_im, height, width, ksize_h, ksize_w,
      pad_h, pad_w, stride_h, stride_w,
      dilation_h, dilation_w,
      height_col, width_col, data_col
  );
  THCudaCheck(cudaGetLastError());
}

/*** Encode/decode region ***/
__forceinline__ __device__ unsigned int encode_val(float* array) {
    unsigned int val = 0;
    unsigned int sign;

    for(int i=0; i<32; i++)
    {
        sign = (array[i]>0);
        val |= (sign<<i);
    }
    return val;
}

__global__ void encode_rows_kernel(float *input, unsigned int* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i<size) output[i] = encode_val(&input[i*32]); 
}

__global__ void encode_cols_kernel(float *a, unsigned int* b, int m, int n) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
	
	int i32 = i*32;
	
    if (j < n && i32 < m) {
        float num;
        unsigned int rvalue = 0;
        unsigned int sign;
	
		#pragma unroll 4
        for(int k = 0; k < 32; k++) {
            num = a[j + n * (i32 + k)];
            sign = (num > 0);
            rvalue |= (sign << k);
        }
        b[j + n * i] = rvalue;
    }
}

extern "C"
void encode_rows(THCState *state, THCudaTensor* input, THCudaIntTensor* output) {
      THCUNN_assertSameGPU(state, 2, input, output);

      int count = THCudaIntTensor_nElement(state, output);
			
      encode_rows_kernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
            THCudaTensor_data(state, input),
            (unsigned int*)THCudaIntTensor_data(state, output),
            count);
}

extern "C"
void encode_cols(THCState *state, THCudaTensor* input, THCudaIntTensor* output) {
      THCUNN_assertSameGPU(state, 2, input, output);

	  int n = input->size[0];
	  int k = input->size[1];
	  
	  dim3 blockDim(32, 32, 1);
      dim3 gridDim(k/32, n/32, 1);
	  
	  	  std::chrono::duration<double> total(0);
			
			//for (int i=0;i<100;i++){
            auto start = std::chrono::high_resolution_clock::now();

      encode_cols_kernel <<< gridDim,blockDim, 0, THCState_getCurrentStream(state) >>>(
            THCudaTensor_data(state, input),
            (unsigned int*)THCudaIntTensor_data(state, output),
            n, k);
    	cudaDeviceSynchronize();
			
	 auto end = std::chrono::high_resolution_clock::now();
			total += end -start;
			//}
            std::chrono::duration<double> diff = total;
            std::cout << "GEMM kernel time: " << diff.count() << " s\n"; 
}

__forceinline__ __device__ void decode_val(unsigned int input, float* output) {
    for (int i=0; i<32; ++i) 
    {
        output[i] = (input & (1 << i)) >> i;
    }
}

__global__ void decode_rows_kernel(unsigned int* input, float *output, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i<size) decode_val(input[i],&output[i*32]);
}

extern "C"
void decode(THCState *state, THCudaIntTensor* input, THCudaTensor* output) {
      THCUNN_assertSameGPU(state, 2, input, output);

      int count = THCudaIntTensor_nElement(state, input);
	  
      decode_rows_kernel <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>(
            (unsigned int*)THCudaIntTensor_data(state, input),
            THCudaTensor_data(state, output),
            count);

      THCudaCheck(cudaGetLastError());
}

// Based on the torch SpatialConvolutionMM_updateOutput
extern "C"
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
           int padW, int padH) {

    THCUNN_assertSameGPU(state, 5, input, output, weight, columns, columns_binary); 

    // Params:
    int nInputPlane = weight->nDimension == 2 ? weight->size[1]/(kH*kW)*32 : weight->size[1];
    int nOutputPlane = weight->size[0];

    input = THCudaTensor_newContiguous(state, input);
    int batch = 1;
    if (input->nDimension == 3) {
        // Force batch
        batch = 0;
        THCudaTensor_resize4d(state, input, 1, input->size[0], input->size[1], input->size[2]);
    }

    long inputWidth   = input->size[3];
    long inputHeight  = input->size[2];
    long outputWidth  = (inputWidth + 2*padW - kW) / dW + 1;
    long outputHeight = (inputHeight + 2*padH - kH) / dH + 1;

    // Batch size + input planes
    long batchSize = input->size[0];

    // Resize output
    THCudaTensor_resize4d(state, output, batchSize, nOutputPlane, outputHeight, outputWidth);
	
    // Resize temporary columns
    THCudaTensor_resize2d(state, columns, nInputPlane*kW*kH, outputHeight*outputWidth); 

	// Resize the weight/columns buffers
	if (columns_binary->nDimension != 2 || columns_binary->size[0]*columns_binary->size[1] < outputHeight*outputWidth*weight->size[1]) {
		THCudaIntTensor_resize2d(state, columns_binary, weight->size[1], outputHeight*outputWidth);
	}
	
    // Helpers
    THCudaTensor *input_n = THCudaTensor_new(state);
    THCudaTensor *output_n = THCudaTensor_new(state);  
	
    // For each elt in batch, do:
    for (int elt = 0; elt < batchSize; elt ++) {
        // Matrix mulitply per output:
        THCudaTensor_select(state, input_n, input, 0, elt);
        THCudaTensor_select(state, output_n, output, 0, elt);
		
        // Extract columns:
        im2col(
            THCState_getCurrentStream(state),
            THCudaTensor_data(state, input_n),
            nInputPlane, inputHeight, inputWidth, kH, kW, padH, padW, dH, dW,
            1, 1, THCudaTensor_data(state, columns)
        );
        
        // M,N,K are dims of matrix A and B
        // (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
        // row-major to column-major change
        int m = weight->size[0];
        int n = weight->size[1];
        int k = columns->size[1];

		// Encode cols
		dim3 blockDim_ck(32, 32, 1);
		dim3 gridDim_ck(columns->size[1]/32+1, columns->size[0]/32+1, 1);

		encode_cols_kernel <<< gridDim_ck, blockDim_ck, 0, THCState_getCurrentStream(state) >>>(
			THCudaTensor_data(state, columns),
			(unsigned int*)THCudaIntTensor_data(state, columns_binary),
			columns->size[0], columns->size[1]);

		dim3 blockDim(16, 16, 1);
		dim3 gridDim(k/16+1 , m/16+1);
			
		// Do here the binary_gemm call - popcount(XOR) is called here
		binary_gemm <<<gridDim,blockDim,0,THCState_getCurrentStream(state)>>>(
            (unsigned int*)THCudaIntTensor_data(state, weight),
            (unsigned int*)THCudaIntTensor_data(state, columns_binary), 
            THCudaTensor_data(state, output_n), m, n, k, THCudaTensor_data(state, alphas));
    }

    // Free
    THCudaTensor_free(state, input_n);
    THCudaTensor_free(state, output_n);
	
    if (batch==0) {
        THCudaTensor_resize3d(state, output, nOutputPlane, outputHeight, outputWidth);
        THCudaTensor_resize3d(state, input, nInputPlane, inputHeight, inputWidth);
    }

    THCudaTensor_free(state,input);
}