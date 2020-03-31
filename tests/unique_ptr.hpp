#include <cassert>
#include <cumem/unique_ptr.hpp>
#include <cumem/allocator/allocator_delete.hpp>
#include <cumem/allocator/allocator_new.hpp>


#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __managed__
#define __managed__
#endif

struct base {};
struct derived : base {};


__managed__ bool deleted;


template<class T>
struct my_deleter
{
  bool* deleted;

  using pointer = T*;

  __host__ __device__
  void operator()(T* ptr)
  {
    cumem::allocator<T> alloc;
    cumem::allocator_delete(alloc, ptr);
    *deleted = true;
  }

  __host__ __device__
  my_deleter(bool* d) : deleted(d) {}

  __host__ __device__
  my_deleter() : my_deleter(nullptr) {}

  template<class U>
  __host__ __device__
  my_deleter(const my_deleter<U>& other) : my_deleter(other.deleted) {}
};


__host__ __device__
void test_constructors()
{
  using namespace cumem;

  {
    // with default_delete

    allocator<int> alloc;
    int expected = 13;

    // pointer ctor
    unique_ptr<int> p0(allocator_new<int>(alloc, expected));
    assert(expected == *p0);

    // null ctor
    unique_ptr<int> p1;
    assert(!p1);

    // move ctor
    unique_ptr<int> p2 = std::move(p0);
    assert(expected == *p2);
    assert(!p0);

    // converting ctor
    unique_ptr<derived> pd;
    unique_ptr<base> pb = std::move(pd);
  }

  {
    // with my_deleter

    allocator<int> alloc;
    int expected = 13;

    deleted = false;
    {
      // pointer ctor
      unique_ptr<int, my_deleter<int>> p0(allocator_new<int>(alloc, expected), my_deleter<int>{&deleted});
      assert(expected == *p0);

      // move ctor
      unique_ptr<int, my_deleter<int>> p2 = std::move(p0);
      assert(expected == *p2);
      assert(!p0);
    }
    assert(deleted);

    // converting ctor
    unique_ptr<derived, my_deleter<derived>> pd(nullptr, my_deleter<derived>{&deleted});
    unique_ptr<base, my_deleter<base>> pb = std::move(pd);
  }
}


struct set_on_delete
{
  bool& deleted;

  __host__ __device__
  set_on_delete(bool& d) : deleted(d) {}

  __host__ __device__
  ~set_on_delete()
  {
    deleted = true;
  }
};


__host__ __device__
void test_destructor()
{
  using namespace cumem;

  {
    // with default_delete
    deleted = false;
    allocator<set_on_delete> alloc;

    {
      unique_ptr<set_on_delete> p0(allocator_new<set_on_delete>(alloc, deleted));
    }

    assert(deleted);
  }
}


__host__ __device__
void test_move_assignment()
{
  using namespace cumem;

  int expected = 13;

  unique_ptr<int> p0 = make_unique<int>(expected);
  unique_ptr<int> p1;
  p1 = std::move(p0);

  assert(expected == *p1);
  assert(!p0);
}


__host__ __device__
void test_get()
{
  using namespace cumem;

  allocator<int> alloc;
  int* expected = allocator_new<int>(alloc);

  unique_ptr<int> p0{expected};
  assert(expected == p0.get());
}


__host__ __device__
void test_release()
{
  using namespace cumem;

  allocator<int> alloc;
  int* expected = allocator_new<int>(alloc);

  unique_ptr<int> p0{expected};
  int* result = p0.release();
  assert(expected == result);
  assert(!p0);

  allocator_delete(alloc, expected);
}


__host__ __device__
void test_reset()
{
  using namespace cumem;

  allocator<int> alloc;

  {
    // reset null ptr

    int* ptr = allocator_new<int>(alloc);
    unique_ptr<int> p;
    p.reset(ptr);

    assert(ptr == p.get());
  }

  {
    // reset non-null ptr

    unique_ptr<int> p{allocator_new<int>(alloc)};

    int* expected = allocator_new<int>(alloc);
    p.reset(expected);

    assert(expected == p.get());
  }
}


template<class T>
__host__ __device__
const T& cref(const T& ref)
{
  return ref;
}


__host__ __device__
void test_get_deleter()
{
  cumem::unique_ptr<int> p;

  p.get_deleter();
  cref(p).get_deleter();
}


__host__ __device__
void test_operator_star()
{
  using namespace cumem;

  int expected = 13;

  unique_ptr<int> p = make_unique<int>(13);
  assert(expected == *p);
  assert(expected == *cref(p));
}


__host__ __device__
void test_operator_bool()
{
  using namespace cumem;

  unique_ptr<int> null;
  assert(!null);

  unique_ptr<int> p = make_unique<int>(13);
  assert(bool(p));
}


__host__ __device__
void test_swap()
{
  using namespace cumem;

  int expected1 = 13;
  int expected2 = 7;

  unique_ptr<int> p1 = make_unique<int>(expected1);
  unique_ptr<int> p2 = make_unique<int>(expected2);

  p1.swap(p2);

  assert(expected1 == *p2);
  assert(expected2 == *p1);
}


#ifdef __CUDACC__
template<class F>
__global__ void device_invoke(F f)
{
  f();
}
#endif


void test_unique_ptr()
{
  test_constructors();
  test_destructor();
  test_move_assignment();
  test_get();
  test_release();
  test_reset();
  test_get_deleter();
  test_operator_star();
  test_operator_bool();

#ifdef __CUDACC__
  device_invoke<<<1,1>>>([] __device__ ()
  {
    test_constructors();
    test_destructor();
    test_move_assignment();
    test_get();
    test_release();
    test_reset();
    test_get_deleter();
    test_operator_star();
    test_operator_bool();
  });
  assert(cudaDeviceSynchronize() == cudaSuccess);
#endif
}

