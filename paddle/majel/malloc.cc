#include "malloc.h"
#include <glog/logging.h>
#include "cuda_device.h"

namespace majel {
namespace malloc {
namespace detail {

class DefaultAllocator {
public:
  static void* malloc(majel::Place place, size_t size);

  static void free(majel::Place, void* ptr);
};

class DefaultAllocatorMallocVisitor : public boost::static_visitor<void*> {
public:
  DefaultAllocatorMallocVisitor(size_t size) : size_(size) {}

  void* operator()(majel::CpuPlace p) {
    void* address;
    CHECK_EQ(posix_memalign(&address, 32ul, size_), 0);
    CHECK(address) << "Fail to allocate CPU memory: size=" << size_;
    return address;
  }

#ifndef PADDLE_ONLY_CPU
  void* operator()(majel::GpuPlace p) {
    void* address = majel::gpu::detail::malloc(size_);
    CHECK(address) << "Fail to allocate GPU memory " << size_ << " bytes";
    return address;
  }
#else
  void* operator()(majel::GpuPlace p) {
    CHECK(majel::is_cpu_place(p)) << "GPU Place not supported";
    return nullptr;
  }
#endif

private:
  size_t size_;
};

class DefaultAllocatorFreeVisitor : public boost::static_visitor<void> {
public:
  DefaultAllocatorFreeVisitor(void* ptr) : ptr_(ptr) {}
  void operator()(majel::CpuPlace p) {
    if (ptr_) {
      ::free(ptr_);
    }
  }

#ifndef PADDLE_ONLY_CPU
  void operator()(majel::GpuPlace p) {
    if (ptr_) {
      majel::gpu::detail::free(ptr_);
    }
  }

#else
  void operator()(majel::GpuPlace p) {
    CHECK(majel::is_cpu_place(p)) << "GPU Place not supported";
  }
#endif

private:
  void* ptr_;
};

void* DefaultAllocator::malloc(majel::Place place, size_t size) {
  DefaultAllocatorMallocVisitor visitor(size);
  return boost::apply_visitor(visitor, place);
}

void DefaultAllocator::free(majel::Place place, void* ptr) {
  DefaultAllocatorFreeVisitor visitor(ptr);
  boost::apply_visitor(visitor, place);
}

}  // namespace detail
}  // namespace malloc
}  // namespace majel
namespace majel {
namespace malloc {

void* malloc(majel::Place place, size_t size) {
  return detail::DefaultAllocator::malloc(place, size);
}

void free(majel::Place place, void* ptr) {
  detail::DefaultAllocator::free(place, ptr);
}
}  // namespace malloc
}  // namespace majel
