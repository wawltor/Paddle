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

#include <string>
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/gather_op.h"
#include "paddle/fluid/operators/scatter.cu.h"
#include "paddle/fluid/operators/scatter_op.h"

namespace paddle {
namespace operators {

template <typename T>
class ScatterOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *X = ctx.Input<Tensor>("X");
    auto *Ids = ctx.Input<Tensor>("Ids");
    auto *Updates = ctx.Input<Tensor>("Updates");
    auto *Out = ctx.Output<Tensor>("Out");
    std::string mode = ctx.Attr<std::string>("mode");

    Out->ShareDataWith(*X);

    // use template class to support index32_t and index64_t
    const auto &index_type = Ids->type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE(
        index_type_match,
        "Index holds the wrong type, it holds %s, but desires to be %s or %s",
        paddle::framework::DataTypeToString(index_type),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT64));
    if (index_type == framework::proto::VarType::INT32) {
      GPUScatterAssign<T, int32_t>(ctx, *Updates, *Ids, Out, mode);
    } else {
      GPUScatterAssign<T, int64_t>(ctx, *Updates, *Ids, Out, mode);
    }
  }
};

template <typename T>
class ScatterGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");
    auto *dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *dUpdates = ctx.Output<Tensor>(framework::GradVarName("Updates"));
    auto *Ids = ctx.Input<Tensor>("Ids");
    auto *dOut = ctx.Input<Tensor>(framework::GradVarName("Out"));
    if (dX) {
      // In place gradient: dX = dO
      framework::TensorCopy(*dOut, ctx.GetPlace(), dX);
    }
    if (dUpdates) {
      dUpdates->mutable_data<T>(ctx.GetPlace());
    }

    const auto &index_type = Ids->type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE(
        index_type_match,
        "Index holds the wrong type, it holds %s, but desires to be %s or %s",
        paddle::framework::DataTypeToString(index_type),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT32),
        paddle::framework::DataTypeToString(framework::proto::VarType::INT64));
    // Gradient by Gather: dUpdates = dO[Ids]
    if (index_type == framework::proto::VarType::INT32) {
      GPUGather<T, int32_t>(ctx.device_context(), *dOut, *Ids, dUpdates);
    } else {
      GPUGather<T, int64_t>(ctx.device_context(), *dOut, *Ids, dUpdates);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(scatter, ops::ScatterOpCUDAKernel<float>);
REGISTER_OP_CUDA_KERNEL(scatter_grad, ops::ScatterGradOpCUDAKernel<float>);
