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
#include <memory.h>
#include <cstring>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/jit/kernels.h"

namespace paddle {
namespace operators {

using framework::Tensor;

/**
 * A thin wrapper for gathering on cpu tensor
 * Return a new tensor from source tensor, gathered according to index
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void CPUGather(const platform::DeviceContext& ctx, const Tensor& src,
               const Tensor& index, Tensor* output) {
  PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()));
  // check index of shape 1-D
  PADDLE_ENFORCE(index.dims().size() == 1 ||
                 (index.dims().size() == 2 && index.dims()[1] == 1));
  int64_t index_size = index.dims()[0];

  auto src_dims = src.dims();

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // slice size
  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t slice_bytes = slice_size * sizeof(T);

  for (int64_t i = 0; i < index_size; ++i) {
    IndexT index_ = p_index[i];
    memcpy(p_output + i * slice_size, p_src + index_ * slice_size, slice_bytes);
  }
}

template <typename T, typename IndexT = int>
void CPUGatherAdd(const framework::ExecutionContext &context, const Tensor* src,
               const Tensor* index, const int& first_index, const int& last_index,
               Tensor* output, int& out_index) {
  PADDLE_ENFORCE(platform::is_cpu_place(context.device_context().GetPlace()));
  // check index of shape 1-D
  PADDLE_ENFORCE(index->dims().size() == 1 ||
                 (index->dims().size() == 2 && index->dims()[1] == 1));
  int64_t index_size = index->dims()[0];

  // the col_index
  PADDLE_ENFORCE((last_index >= first_index) && (last_index <= index_size -1));

  auto src_dims = src->dims();

  const T* p_src = src->data<T>();
  const IndexT* p_index = index->data<IndexT>();
  T* out_table = output->data<T>();

  // table width
  int table_width = 1;
  for (int i = 1; i < src_dims.size(); ++i) table_width *= src_dims[i];
  //table height
  int table_height = 1;
  table_height = src_dims[0];

  PADDLE_ENFORCE(out_index < table_height);

  memset(out_table + out_index * table_width, 0, sizeof(T) * table_width);
  auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
  //auto compute = jit::KernelFuncs<jit::VAddTuple<T>, platform::CPUPlace>::Cache().At(first_index);
  for (int64_t i = first_index; i <= last_index; ++i) {
    IndexT index_ = p_index[i];
    ///compute(p_src + index_ * table_width, out_table + out_index * table_width, 
    //  out_table + out_index * table_width, table_width);
    //use blas lib to execute add op
    blas.VADD(table_width, p_src + index_ * table_width, out_table + out_index * table_width,
        out_table + out_index * table_width);
  }
}

}  // namespace operators
}  // namespace paddle
