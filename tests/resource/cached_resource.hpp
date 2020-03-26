#include <cassert>
#include <cumem/resource/cached_resource.hpp>
#include <cumem/resource/managed_resource.hpp>
#include <limits>
#include <utility>


void test_allocate()
{
  using cached_resource = cumem::cached_resource<cumem::managed_resource>;

  cached_resource r{0};

  int* ptr = static_cast<int*>(r.allocate(sizeof(int)));

  cudaPointerAttributes attr{};
  assert(cudaPointerGetAttributes(&attr, ptr) == cudaSuccess);
  assert(attr.type == cudaMemoryTypeManaged);
  assert(attr.device == r.base_resource().device());
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
  using cached_resource = cumem::cached_resource<cumem::managed_resource>;

  cached_resource r0{0};
  cached_resource r1{1};

  // same resource compares same
  assert(r0.is_equal(r0));
  assert(r0 == r0);
  assert(!(r0 != r0));

  // different resources compare different
  assert(!r0.is_equal(r1));
  assert(r0 != r1);

  // resources pointing to same device compare different
  cached_resource other_r0{0};
  assert(!r0.is_equal(other_r0));
  assert(!(r0 == other_r0));
  assert(r0 != other_r0);
}


void test_move_construction()
{
  using cached_resource = cumem::cached_resource<cumem::managed_resource>;

  cached_resource r0;

  void* ptr = r0.allocate(1);

  cached_resource moved_r0 = std::move(r0);

  moved_r0.deallocate(ptr, 1);
}


void test_throw_on_failure()
{
  using cached_resource = cumem::cached_resource<cumem::managed_resource>;

  cached_resource r;

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


void test_cached_resource()
{
  test_allocate();
  test_comparison();
  test_move_construction();
  test_throw_on_failure();
}

