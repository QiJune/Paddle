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
struct mul {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& input0,
                  const framework::Tensor& input1,
                  framework::Tensor* output) {
    int size = 4;
    float* t_a = (float*)malloc(size * sizeof(float));
    float* t_b = (float*)malloc(size * sizeof(float));
    cudaMemcpy(t_a,
               input0.data<float>(),
               size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(t_b,
               input1.data<float>(),
               size * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
      std::cout << t_a[i] << " " << t_b[i] << std::endl;
    }

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair = {
        {Eigen::IndexPair<Eigen::DenseIndex>(1, 0)}};

    framework::EigenMatrix<T>::From(*output).device(
        *(device_context.get_eigen_device<Place>())) =
        framework::EigenMatrix<T>::From(input0).contract(
            framework::EigenMatrix<T>::From(input1), dim_pair);

    float* t_c = (float*)malloc(size * sizeof(float));
    cudaMemcpy(t_c,
               output->data<float>(),
               size * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; i++) {
      std::cout << t_c[i] << std::endl;
    }
  }
};
}  // namespace functors
}  // namespace operators
}  // namespace paddle
