#include <cassert>
#include <cumem/resource/managed_resource.hpp>
#include <limits>


void test_allocate()
{
  cumem::managed_resource r;

  int* ptr = static_cast<int*>(r.allocate(sizeof(int)));

  cudaPointerAttributes attr{};
  assert(cudaPointerGetAttributes(&attr, ptr) == cudaSuccess);
  assert(attr.type == cudaMemoryTypeManaged);
  assert(attr.device == r.device());
  assert(attr.devicePointer == ptr);
  assert(attr.hostPointer == ptr);

  int expected = 13;
  *ptr = expected;
  int result = *ptr;

  assert(expected == result);

  r.deallocate(ptr, sizeof(int));
}


void test_comparison()
{
  cumem::managed_resource r0{0};
  cumem::managed_resource r1{1};

  // same resource compares same
  assert(r0.is_equal(r0));
  assert(r0 == r0);
  assert(!(r0 != r0));

  // resources pointing to same device compare same
  cumem::managed_resource other_r0{0};
  assert(r0.is_equal(other_r0));
  assert(r0 == other_r0);
  assert(!(r0 != other_r0));

  // resources pointing to different devices compare different
  assert(!r0.is_equal(r1));
  assert(r0 != r1);
}


void test_copy_construction()
{
  cumem::managed_resource r0{0};
  cumem::managed_resource copy = r0;

  assert(r0 == copy);
}


void test_device()
{
  cumem::managed_resource r;
  assert(r.device() == 0);

  cumem::managed_resource r1{1};
  assert(r1.device() == 1);
}


void test_throw_on_failure()
{
  cumem::managed_resource r;

  try
  {
    std::size_t num_bytes = std::numeric_limits<std::size_t>::max();
    r.allocate(num_bytes);
  }
  catch(...)
  {
    return;
  }

  assert(0);
}


void test_managed_resource()
{
  test_allocate();
  test_comparison();
  test_copy_construction();
  test_device();
  test_throw_on_failure();
}

