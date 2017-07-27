#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/tensor.h"
#include "paddle/platform/device_context.h"

namespace paddle {
namespace operators {
namespace functors {

template <typename Place, typename T>
struct softmax {
  void operator()(const platform::DeviceContext& device_context,
                  const framework::Tensor& input,
                  framework::Tensor* output) {
    auto logits = framework::EigenMatrix<T>::From(input);
    auto softmax = framework::EigenMatrix<T>::From(*output);

    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);

    auto shifted_logits = (logits -
                           logits.maximum(along_class)
                               .eval()
                               .reshape(batch_by_one)
                               .broadcast(one_by_class));

    softmax.device(*(device_context.get_eigen_device<Place>())) =
        shifted_logits.exp();

    softmax.device(*(device_context.get_eigen_device<Place>())) =
        (softmax *
         softmax.sum(along_class)
             .inverse()
             .eval()
             .reshape(batch_by_one)
             .broadcast(one_by_class));
  }
};

}  // namespace functors
}  // namespace operators
}  // namespace paddle
