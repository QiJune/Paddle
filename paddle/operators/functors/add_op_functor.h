#pragma once
#ifndef PADDLE_ONLY_CPU
#define EIGEN_USE_GPU
#endif
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"
namespace paddle {
namespace operators {
namespace functors {

template <typename Place, typename T>
struct add {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& input1,
                  const framework::Tensor& input2,
                  framework::Tensor* output) {
    framework::EigenVector<T>::Flatten(*output).device(
        *(device_context.get_eigen_device<Place>())) =
        framework::EigenVector<T>::Flatten(input1) +
        framework::EigenVector<T>::Flatten(input2);
  }
};

}  // namespace functors
}  // namespace operators
}  // namespace paddle
