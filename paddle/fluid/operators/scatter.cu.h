/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <unordered_set>
#include "math/math_function.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/operators/scatter_op.h"
#include "cuda.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T, typename IndexT = int32_t>
__global__ void ScatterInitCUDAKernel(const IndexT* indices, T* output,
                                      size_t index_size, size_t slice_size,
                                      bool overwrite) {
  CUDA_1D_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];
    IndexT out_i = scatter_i * slice_size + slice_i;
    *(output + out_i) = static_cast<T>(0);
  }
}

template <typename T, typename IndexT = int32_t>
__global__ void ScatterCUDAKernel(const T* params, const IndexT* indices,
                                  T* output, size_t index_size,
                                  size_t slice_size) {
  CUDA_1D_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];
    IndexT out_i = scatter_i * slice_size + slice_i;
    *(output + out_i) = *(params + i);
  }
}

template <typename T, typename IndexT = int32_t>
__global__ void ScatterAddCUDAKernel(const T* params, const IndexT* indices,
                                  T* output, size_t index_size,
                                  size_t slice_size) {
  CUDA_1D_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];
    IndexT out_i = scatter_i * slice_size + slice_i;
    paddle::platform::CudaAtomicAdd(output + out_i, *(params + i));
  }
}


template <typename T, typename IndexT = int32_t>
__global__ void ScatterMaxCUDAKernel(const T* params, const IndexT* indices,
                                  T* output, size_t index_size,
                                  size_t slice_size) {
  CUDA_1D_KERNEL_LOOP(i, index_size * slice_size) {
    int indices_i = i / slice_size;
    int slice_i = i - indices_i * slice_size;  // offset inside the slice
    IndexT scatter_i = indices[indices_i];
    IndexT out_i = scatter_i * slice_size + slice_i;
    //paddle::platform::CudaAtomicAdd(output + out_i, *(params + i));
  }
}

// use the enable_if to get different function in different template
template<typename T>
__device__ typename std::enable_if<std::is_floating_point<T>::value>::type 
AutomicMax(T* output, T value) {
	 atomicMax(output, value);
}

template<typename T>
__device__ typename std::enable_if<!std::is_floating_point<T>::value>::type 
AutomicMax(T* output, T value) {
	T* output_int = (T*) output;
	T old = *output_int, assumed;
	do {
		assumed = old;
		old = atomicCAS(output_int, assumed,
				__float_as_int(max(value, __int_as_float(assumed))));
	} while (assumed != old);
}
/*
template<typename T>
__device__ typename std::enable_if(!std::is_floating_point<T>::value)::type 
AutomicMax(T*output, T value) {
	T* src_int = (T*) src;
	T old = *src_t, assumed;
	do {
		assumed = old;
		old = atomicCAS(src_int, assumed,
				__float_as_int(max(value, __int_as_float(assumed))));
	} while (assumed != old);
}*/

/**
 * A thin wrapper on gpu tensor
 * Return a new updated tensor from source tensor, scatter-assigned according to
 * index
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT>
void GPUScatterAssign(const framework::ExecutionContext& context,
                      const Tensor& src, const Tensor& index, Tensor* output,
                      std::string mode) {
  // PADDLE_ENFORCE(platform::is_gpu_place(place));
  // check index of shape 1-D

  const auto& ctx = context.device_context();
  PADDLE_ENFORCE(index.dims().size() == 1 ||
                 (index.dims().size() == 2 && index.dims()[1] == 1));
  int index_size = index.dims()[0];

  auto src_dims = src.dims();
  framework::DDim output_dims(src_dims);
  output_dims[0] = index_size;

  // slice size
  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];


  const T* p_src = src.data<T>();
  const auto* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();
  const size_t& slice_bytes = slice_size * sizeof(T);

  // set block and grid num
  int block = 512;
  int n = slice_size * index_size;
  int grid = (n + block - 1) / block;
  
	if (mode == MODE_OVERWRITE) {
			ScatterCUDAKernel<T, IndexT><<<
				grid, block, 0,
				reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
						p_src, p_index, p_output, index_size, slice_size);
	} else if (mode == MODE_ADD) {
			ScatterAddCUDAKernel<T, IndexT><<<
				grid, block, 0,
				reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
						p_src, p_index, p_output, index_size, slice_size);
	} else if (mode == MODE_MAX) {
			ScatterMaxCUDAKernel<T, IndexT><<<
				grid, block, 0,
				reinterpret_cast<const platform::CUDADeviceContext&>(ctx).stream()>>>(
						p_src, p_index, p_output, index_size, slice_size);
	}
}

}  // namespace operators
}  // namespace paddle
