#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/operators/functors/add_op_functor.h"
#include "paddle/operators/functors/mul_op_functor.h"
#include "paddle/operators/functors/softmax_op_functor.h"

#include <cmath>
#include <iostream>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "unsupported/Eigen/CXX11/Tensor"

using namespace paddle::framework;
using namespace paddle::platform;
using namespace paddle::operators;

template <typename T>
using Matrix =
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, Eigen::DenseIndex>,
                     Eigen::Aligned>;
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
TEST(OpFunctor, MulCPU) {
  int size = 4;
  float* t_a = (float*)malloc(size * sizeof(float));
  float* t_b = (float*)malloc(size * sizeof(float));
  float* t_c = (float*)malloc(size * sizeof(float));
  for (int i = 0; i < size; i++) {
    t_a[i] = i;
    t_b[i] = i;
  }
  Tensor t1;
  t1.mutable_data<float>({2, 2}, CPUPlace());
  Tensor t2;
  t2.mutable_data<float>({2, 2}, CPUPlace());
  Tensor t3;
  t3.mutable_data<float>({2, 2}, CPUPlace());
  std::memcpy(t1.data<float>(), t_a, size * sizeof(float));
  std::memcpy(t2.data<float>(), t_b, size * sizeof(float));
  functors::mul<CPUPlace, float> functor;
  DeviceContext* device = new CPUDeviceContext();
  functor(*device, t1, t2, &t3);
  std::memcpy(t_c, t3.data<float>(), size * sizeof(float));
  EXPECT_EQ(t_c[0], 2);
  EXPECT_EQ(t_c[1], 3);
  EXPECT_EQ(t_c[2], 6);
  EXPECT_EQ(t_c[3], 11);
}
TEST(OpFunctor, SoftmaxCPU) { EXPECT_EQ(1, 1); }
