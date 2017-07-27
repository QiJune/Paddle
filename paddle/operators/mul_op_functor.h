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
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair = {
        {Eigen::IndexPair<Eigen::DenseIndex>(1, 0)}};

    framework::EigenMatrix<T>::From(*output).device(
        *(device_context.get_eigen_device<Place>())) =
        framework::EigenMatrix<T>::From(input0).contract(
            framework::EigenMatrix<T>::From(input1), dim_pair);
  }
};
}  // namespace functors
}  // namespace operators
}  // namespace paddle
