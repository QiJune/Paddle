#include "paddle/majel/array.h"
#include <gtest/gtest.h>

#include <vector>

__global__ void set_gpu(majel::ArrayView<float, 2> t, majel::Dim<2> idx, float v) {
  t[idx] = v;
}


void TestArray(void) {
  // Construct array on CPU
  using namespace majel;

  Dim<2> idx(1,1);
  Array<float, 2> the_array(Dim<2>(2, 3), GpuPlace());
  //Can I assign to Arrays on the GPU?
  the_array[idx] = 3.0f;
  ASSERT_EQUAL(3.0f, get(the_array, idx));


  the_array = Array<float, 2>(Dim<2>(2, 3), CpuPlace());
  // Flatten
  Array<float, 1> flat_arr = flatten(the_array);

  //Can I assign to Arrays on the CPU?
  the_array[idx] = 2.0f;
  ASSERT_EQUAL(2.0f, get(the_array, idx));

  Dim<2> min_idx(0, 0);
  Dim<2> max_idx(1, 2);

  // Set and read elements
  the_array[min_idx] = 5;
  ASSERT_EQUAL(5, the_array[min_idx]);

  the_array[max_idx] = 2;
  ASSERT_EQUAL(2, get(the_array, max_idx));

  set(the_array, make_dim(0, 1), 3.0f);
  ASSERT_EQUAL(3, the_array[make_dim(0, 1)]);




  //Can I print a 1D Array
  {
    Dim<1> idx(4);
    Array<float, 1> array(idx, CpuPlace());

    for (int row = 0; row < 4; ++row) {
        Dim<1> dim(row);
        array[dim] = row;
    }

    std::stringstream ss;
    ss << array << std::endl;
    ASSERT_EQUAL(ss.str(), "         0 \n         1 \n         2 \n         3 \n\n");
  }

  //Can I print a 2D Array from GPU
  {
    Dim<2> idx(2, 3);
    Array<float, 2> array(idx, GpuPlace());
    for (int row = 0; row < 2; ++row) {
      for (int col = 0; col < 3; ++col) {
        Dim<2> dim(row, col);
        array[dim] = col;
      }
    }

    std::stringstream ss;
    ss << array << std::endl;
    ASSERT_EQUAL(ss.str(), "         0          1          2 \n         0          1          2 \n\n");
  }

  //Test make_array
  {
    std::vector<int> tvec = {1, 0, -20, 29, 2};

    Array<int, 1> tarray = make_array(tvec);

    ASSERT_EQUAL(majel::get<0>(tarray.size()), tvec.size());
    ASSERT_EQUAL(is_cpu_place(tarray.place()), true);
    ASSERT_EQUAL(is_gpu_place(tarray.place()), false);

    ASSERT_EQUAL(tarray[make_dim(0)], 1);
    ASSERT_EQUAL(tarray[make_dim(1)], 0);
    ASSERT_EQUAL(tarray[make_dim(2)], -20);
    ASSERT_EQUAL(tarray[make_dim(3)], 29);
    ASSERT_EQUAL(tarray[make_dim(4)], 2);

    std::vector<float> gvec = {0.01f, 3.6f, 0.0f, 20.5f, 2.53f, -700.0f};
    Array<float, 1> garray = make_array(gvec, GpuPlace());
    ASSERT_EQUAL(majel::get<0>(garray.size()), gvec.size());
    ASSERT_EQUAL(is_cpu_place(garray.place()), false);
    ASSERT_EQUAL(is_gpu_place(garray.place()), true);
  }
}

TEST(Array, construct) {
  TestArray();
}