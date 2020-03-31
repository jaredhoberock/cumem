#include <cassert>
#include <cumem/allocator/allocation_deleter.hpp>
#include <cumem/allocator/allocator.hpp>
#include <limits>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __managed__
#define __managed__
#endif


__host__ __device__
void test_constructors()
{
  using namespace cumem;

  allocator<int> expected;

  allocation_deleter<allocator<int>> allocator_constructed{expected};
  assert(expected == allocator_constructed.allocator());

  allocation_deleter<allocator<int>>  copy_constructed{allocator_constructed};
  assert(expected == copy_constructed.allocator());

  allocation_deleter<allocator<void>> converting_copy_constructed{allocator_constructed};
  assert(expected == converting_copy_constructed.allocator());
}



struct set_on_delete
{
  bool& deleted;

  __host__ __device__
  ~set_on_delete()
  {
    deleted = true;
  }
};


__managed__ bool deleted;


__host__ __device__
void test_call_operator()
{
  using namespace cumem;

  allocator<set_on_delete> alloc;
  allocation_deleter<allocator<set_on_delete>> deleter{alloc};

  set_on_delete* ptr = alloc.allocate(1);

  deleted = false;
  new(ptr) set_on_delete{deleted};

  deleter(ptr);

  assert(deleted = true);
}


template<class T>
class stateful_allocator : public cumem::allocator<T>
{
  public:
    stateful_allocator(const stateful_allocator&) = default;

    __host__ __device__
    stateful_allocator(int state) : state_(state) {}
  
    __host__ __device__
    bool operator==(const stateful_allocator& other) const
    {
      return state_ == other.state_;
    }
  
  private:
    int state_;
};


__host__ __device__
void test_swap()
{
  using namespace cumem;

  stateful_allocator<set_on_delete> alloc0{0}, alloc1{1};

  allocation_deleter<stateful_allocator<set_on_delete>> deleter_a(alloc0);
  allocation_deleter<stateful_allocator<set_on_delete>> deleter_b(alloc1);

  deleter_a.swap(deleter_b);

  assert(alloc1 == deleter_a.allocator());
  assert(alloc0 == deleter_b.allocator());
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_allocation_deleter()
{
  test_call_operator();
  test_constructors();
  test_swap();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test_call_operator();
    test_constructors();
    test_swap();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}


