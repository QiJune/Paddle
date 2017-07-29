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
    int size = 4;
    //  Eigen::CudaStreamDevice sd;
    // Eigen::GpuDevice dd(&sd);
    /*  framework::EigenVector<T>::Flatten(*output).device(
          *(device_context.get_eigen_device<Place>())) =
          framework::EigenVector<T>::Flatten(input1) +
          framework::EigenVector<T>::Flatten(input2);
  */

    float* t_a = (float*)malloc(size * sizeof(float));
    float* t_b = (float*)malloc(size * sizeof(float));
    cudaMemcpy(t_a,
               input1.data<float>(),
               size * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(t_b,
               input2.data<float>(),
               size * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < 4; i++) {
      std::cout << t_a[i] << " " << t_b[i] << std::endl;
    }
    LOG(INFO) << device_context.get_eigen_device<Place>();
    LOG(INFO) << device_context.get_eigen_device<Place>()->ok();
    framework::EigenVector<T>::Flatten(*output).device(
        *(device_context.get_eigen_device<Place>())) =
        framework::EigenVector<T>::Flatten(input1) +
        framework::EigenVector<T>::Flatten(input2);

    /*framework::EigenVector<T>::Flatten(output).device(
         dd) =
         framework::EigenVector<T>::Flatten(input1) +
         framework::EigenVector<T>::Flatten(input2);
 */

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
