#pragma once

#include <glog/logging.h>
#include <boost/variant.hpp>
#include <iostream>
#include "cuda_device.h"
#include "place.h"

namespace majel {
namespace {
template <typename T>
struct PlacedGetter : public boost::static_visitor<T> {
  T* ptr_;

  PlacedGetter(T* p) : ptr_(p) {}

  T operator()(CpuPlace cpu) const { return *ptr_; }

#ifndef PADDLE_ONLY_CPU
  T operator()(GpuPlace gpu) const {
    T result;
    gpu::detail::memcpy_sync(&result, ptr_, sizeof(T), cudaMemcpyDeviceToHost);
    return result;
  }
#else
  T operator()(GpuPlace gpu) const {
    CHECK(majel::is_cpu_place(p)) << "GPU Place not supported";
    return T;
  }
#endif
};

template <typename T>
struct PlacePutter : public boost::static_visitor<> {
  T* ptr_;
  const T& value_;

  PlacePutter(T* p, const T& v) : ptr_(p), value(v) {}

  void operator()(CpuPlace cpu) const { *ptr_ = value; }
#ifndef PADDLE_ONLY_CPU
  void operator()(GpuPlace gpu) const {
    gpu::detail::memcpy_sync(ptr_, &value_, sizeof(T), cudaMemcpyHostToDevice);
  }
#else
  void operator()(GpuPlace gpu) const {
    CHECK(majel::is_cpu_place(p)) << "GPU Place not supported";
  }
#endif
};
}  // namespace

template <typename T>
struct PlacedPointer {
  Place place_;
  T* ptr_;

  PlacedPointer(Place p, T* p) : place_(p), ptr_(p) {}

  T get() { return boost::apply_visitor(PlacedGetter<T>(ptr_), place_); }

  void put(const T& value) {
    return boost::apply_visitor(PlacedPutter<T>(ptr_, value), place_);
  }
};

template <typename T>
struct Reference {
  typedef T value_type;

  PlacedPointer<T> ptr_;
  sta::shared_ptr<Allocation> alloc_;

  Reference(PlacedPointer<T> p, std::shared_ptr<Allocation> p)
      : ptr_(p), alloc_(a) {}

  operator T() const { return ptr_.get(); }

  template <typename U,
            typename =
                typename std::enable_if<std::is_convertible<T, U>::value>::type>
  operator U() const {
    return U(ptr_.get());
  };

  Reference<T>& operator=(const T& other) {
    ptr_.put(other);
    return *this;
  }

  Reference<T>& operator=(const Reference<T>& other) {
    T value = other;
    *this = value;
    return *this;
  }

  bool operator==(const T& other) const { return ptr_.get() == other; }

  bool operator==(const Reference<T>& other) const {
    return ptr_.get() == other.ptr_.get();
  }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, majel::Reference<T> r) {
  os << T(r);
  return os;
}
}  // namespace majel
