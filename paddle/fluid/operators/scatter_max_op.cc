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

#include "paddle/fluid/operators/scatter_max_op.h"
#include <memory>
#include "paddle/fluid/framework/ddim.h"

namespace paddle {
namespace operators {

class ScatterMaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ScatterMaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Ids"),
                   "Input(Ids) of ScatterMaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Updates"),
                   "Input(Updates) of ScatterMaxOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ScatterMaxOp should not be null.");

    auto updates_dims = ctx->GetInputDim("Updates");
    auto ref_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Ids").size(), 1,
                      "Update Ids should be 1-D.");
    PADDLE_ENFORCE_EQ(ref_dims.size(), updates_dims.size(),
                      "Xerence and Updates should have the same shape size");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Updates")[0],
                      ctx->GetInputDim("Ids")[0],
                      "Updates and Ids should have same batch-size.");
    ctx->SetOutputDim("Out", ref_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class ScatterMaxGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    if (ctx->HasOutput(framework::GradVarName("Updates"))) {
      ctx->SetOutputDim(framework::GradVarName("Updates"),
                        ctx->GetInputDim("Updates"));
    }
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"),
                        ctx->GetInputDim(framework::GradVarName("Out")));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
        ctx.device_context());
  }
};

class ScatterMaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of scatter op");
    AddInput("Ids", "The index input of scatter op where X will be updated");
    AddInput("Updates", "The updated value of scatter op");
    AddOutput("Out", "The output of scatter op");
    AddComment(R"DOC(
Scatter Operator.

This operator obtains output by updating the input on selected indices on the first axis, 
and use `max` function to select value between raw inputs and updates.

$$
Out = X \\
Out[Ids] = max(X[Ids], Updates)
$$

)DOC");
  }
};

class ScatterMaxGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("scatter_max_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Ids", Input("Ids"));
    op->SetInput("Updates", Input("Updates"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Updates"), InputGrad("Updates"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(ScatterMaxGradNoNeedBufferVarsInference,
                                      "Updates");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(scatter_max, ops::ScatterMaxOp, ops::ScatterMaxOpMaker,
                  ops::ScatterMaxGradDescMaker);
REGISTER_OPERATOR(scatter_max_grad, ops::ScatterMaxGradOp,
                  ops::ScatterMaxGradNoNeedBufferVarsInference);
REGISTER_OP_CPU_KERNEL(scatter_max, ops::ScatterMaxOpKernel<float>);
REGISTER_OP_CPU_KERNEL(scatter_max_grad, ops::ScatterMaxGradientOpKernel<float>);
