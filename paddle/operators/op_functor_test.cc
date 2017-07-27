#include "gtest/gtest.h"
#include "paddle/operators/add_op_functor.h"
#include "paddle/operators/mul_op_functor.h"
#include "paddle/operators/softmax_op_functor.h"

using namespace paddle::framework;
using namespace paddle::platform;
using namespace paddle::operators;

TEST(OpFunctor, AddCPU) {
  int size = 4;

  float* t_a = (float*)malloc(size * sizeof(float));
  float* t_b = (float*)malloc(size * sizeof(float));
  float* t_c = (float*)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    t_a[i] = i;
    t_b[i] = i;
  }

  Tensor t1;
  t1.mutable_data<float>({4}, CPUPlace());

  Tensor t2;
  t2.mutable_data<float>({4}, CPUPlace());

  Tensor t3;
  t3.mutable_data<float>({4}, CPUPlace());

  std::memcpy(t1.data<float>(), t_a, size * sizeof(float));
  std::memcpy(t2.data<float>(), t_b, size * sizeof(float));

  functors::add<CPUPlace, float> functor;

  DeviceContext* device = new CPUDeviceContext();

  functor(*device, t1, t2, &t3);

  std::memcpy(t_c, t3.data<float>(), size * sizeof(float));

  EXPECT_EQ(t_c[0], 0);
  EXPECT_EQ(t_c[1], 2);
  EXPECT_EQ(t_c[2], 4);
  EXPECT_EQ(t_c[3], 6);
}

TEST(OpFunctor, Mul) { EXPECT_EQ(1, 1); }

TEST(OpFunctor, Softmax) { EXPECT_EQ(1, 1); }

#ifndef PADDLE_ONLY_CPU
TEST(OpFunctor, AddGPU) {
  int size = 4;

  float* t_a = (float*)malloc(size * sizeof(float));
  float* t_b = (float*)malloc(size * sizeof(float));
  float* t_c = (float*)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    t_a[i] = i;
    t_b[i] = i;
  }

  Tensor t1;
  t1.mutable_data<float>({4}, GPUPlace(0));

  Tensor t2;
  t2.mutable_data<float>({4}, GPUPlace(0));

  Tensor t3;
  t3.mutable_data<float>({4}, GPUPlace(0));

  cudaMemcpy(
      t1.data<float>(), t_a, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(
      t2.data<float>(), t_b, size * sizeof(float), cudaMemcpyHostToDevice);

  functors::add<GPUPlace, float> functor;

  DeviceContext* device = new CUDADeviceContext(0);

  functor(*device, t1, t2, &t3);

  cudaMemcpy(
      t_c, t3.data<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_EQ(t_c[0], 0);
  EXPECT_EQ(t_c[1], 2);
  EXPECT_EQ(t_c[2], 4);
  EXPECT_EQ(t_c[3], 6);
}
#endif