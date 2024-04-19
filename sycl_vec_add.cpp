#include <sycl/sycl.hpp>
#include <condition_variable>
#include <mutex>
#include <barrier>
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/global_control.h"
#include "oneapi/tbb/task_arena.h"
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <memory>
#include <syncstream>

using namespace sycl;
typedef std::vector<int> IntVector;

void VectorAdd_tbb(const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel) {
        // A single tbb task arena
        // A workgroup is executed by 1 tbb thread
        // 64 workgroups concurrently executed at maximum

    oneapi::tbb::global_control c(tbb::global_control::max_allowed_parallelism, 64);
    oneapi::tbb::task_arena tbb_arena(64);
    tbb_arena.execute([&] {
           // loop 1
        tbb::parallel_for(tbb::blocked_range<int>(0, sum_parallel.size()),
            [&](tbb::blocked_range<int> r)
            {
                for (int j = r.begin(); j < r.end(); ++j)
                {
                    sum_parallel[j] = a_vector[j] + b_vector[j];
                }
            }, tbb::static_partitioner());
            
            // Implicit barrier 
           // loop 2
        tbb::parallel_for(tbb::blocked_range<int>(0, sum_parallel.size()),
            [&](tbb::blocked_range<int> r)
            {
                for (int j = r.begin(); j < r.end(); ++j)
                {
                    sum_parallel[j] = 5*sum_parallel[j];
                }
            }, tbb::static_partitioner());
    });
}

//************************************
// Vector add in TBB using fixed threads in arena".
//************************************

void VectorAdd_tbb_new_proposal(const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel) {
    // A workgroup is executed by 8 tbb thread in a fixed-size arena
    // 8 workgoups concurrently executed at maximum
    // subgroup size is 16 (SIMD - Platform specific)
    oneapi::tbb::global_control c(tbb::global_control::max_allowed_parallelism, 64);
    const int numThreads= 8;
    const int numArenas = 8;
    std::vector<oneapi::tbb::task_arena> arenas(numArenas);
    //barriers are not copyable
    std::vector<std::barrier<> * > arena_barriers;

    // Initialize the arenas.
    for (int i = 0; i < numArenas; i++) {
        arenas[i].initialize(numThreads, 1);
        arena_barriers.emplace_back(new std::barrier(numThreads));
    }
    auto total_wgs = sum_parallel.size()/128; // 128 is the workgroup size
     // There are total 8 Workgroups each executed in a single fixed arena
     for (int wg =0; wg <total_wgs; wg+=numArenas) {
         
        tbb::parallel_for(tbb::blocked_range<int>(0, numArenas),
        [&](tbb::blocked_range<int> r_wg)
        {
                   for (int i = r_wg.begin(); i < r_wg.end(); ++i)
                   {
                        arenas[r_wg.begin()].execute([&] {
                            // Each workgroup is executed in an arena
                            // and execute 8 subgroups concurrently
                            // r_wg is of size 1
                            tbb::parallel_for(tbb::blocked_range<int>(0, 8),
                                [&](tbb::blocked_range<int> r_subgrp)
                                {
                                    // r_subgrp is of size 1
                                    auto idx_offset = (wg + r_wg.begin())*128 + r_subgrp.begin()*16;
                                    #pragma omp simd
                                    for (int j = 0; j < 16; ++j)
                                    {
                                        auto idx = idx_offset + j;
                                        sum_parallel[idx_offset + j] = a_vector[idx_offset + j] +
                                        b_vector[idx_offset + j];
                                    }
                                    arena_barriers[i]->arrive_and_wait();
                                    #pragma omp simd
                                    for (int j = 0; j < 16; ++j)
                                    {
                                        sum_parallel[idx_offset + j] = 5*sum_parallel[idx_offset + j];
                                    }
                                }, tbb::static_partitioner());
                        });
                   }
        }, tbb::static_partitioner());
     }
}

//************************************
// Vector add in SYCL on device: returns sum in 4th parameter "sum_parallel".
//************************************
void VectorAdd_sycl(queue &q, const IntVector &a_vector, const IntVector &b_vector,
               IntVector &sum_parallel) {
  // Create the range object for the vectors managed by the buffer.
  range<1> num_items{a_vector.size()};

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  buffer a_buf(a_vector);
  buffer b_buf(b_vector);
  buffer sum_buf(sum_parallel.data(), num_items);
  int num_repetitions = 1;
  const int globalSize = a_vector.size(); // Global size
  const int workgroupSize = 8*128; // Workgroup size
  for (size_t i = 0; i < num_repetitions; i++ ) {

    // Submit a command group to the queue by a lambda function that contains the
    // data access permission and device computation (kernel).
    q.submit([&](handler &h) {
      // Create an accessor for each buffer with access permission: read, write or
      // read/write. The accessor is a mean to access the memory in the buffer.
      accessor a(a_buf, h, read_only);
      accessor b(b_buf, h, read_only);
  
      // The sum_accessor is used to store (with write permission) the sum data.
      accessor sum(sum_buf, h, write_only, no_init);
  
      // Use parallel_for to run vector addition in parallel on device. This
      // executes the kernel.
      //    1st parameter is the number of work items.
      //    2nd parameter is the kernel, a lambda that specifies what to do per
      //    work item. The parameter of the lambda is the work item id.
      // SYCL supports unnamed lambda kernel by default.
      // Define the kernel with a specific work group size ?
      h.parallel_for(
           nd_range<1>(globalSize, workgroupSize),
           [=](auto it) {
             // Kernel code here
             auto gid = it.get_global_id(0);
             sum[gid] = a[gid] + b[gid]; 
             it.barrier(access::fence_space::global_space);
             sum[gid] = 5*sum[gid];
           });
    });
  };
  // Wait until compute tasks on GPU done
  q.wait();
}

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

int main(int argc, char* argv[]) {
    auto selector = cpu_selector_v;
    queue q(selector, exception_handler);
    constexpr size_t N = 5120;
    IntVector a(N), b(N);
    IntVector sum_syclParallel(N), sum_tbbParallel(N),
              sum_tbbParallel_nw_proposal(N), sum_sequential(N);
    // fill a and b with random numbers in the unit interval
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100); // Random values between 1 and 100
    std::generate(a.begin(), a.end(), [&]() { return dis(gen); });
    std::generate(b.begin(), b.end(), [&]() { return dis(gen); });
    // zero-out
    std::fill(sum_syclParallel.begin(), sum_syclParallel.end(), 0.0);
    std::fill(sum_tbbParallel.begin(), sum_tbbParallel.end(), 0.0);
    std::fill(sum_tbbParallel_nw_proposal.begin(), sum_tbbParallel_nw_proposal.end(), 0.0);
        // Vector addition in SYCL
    VectorAdd_sycl(q, a, b, sum_syclParallel);

    auto start_VectorAdd_tbb = std::chrono::steady_clock::now();
    VectorAdd_tbb(a, b, sum_tbbParallel);
    auto end_VectorAdd_tbb = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_VectorAdd_tbb = std::chrono::duration_cast<std::chrono::duration<double>>(end_VectorAdd_tbb - start_VectorAdd_tbb);

    auto start_VectorAdd_tbb_new_proposal = std::chrono::steady_clock::now();
    VectorAdd_tbb_new_proposal(a, b, sum_tbbParallel_nw_proposal);
    auto end_VectorAdd_tbb_new_proposal = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration_VectorAdd_tbb_new_proposal = std::chrono::duration_cast<std::chrono::duration<double>>(end_VectorAdd_tbb_new_proposal - start_VectorAdd_tbb_new_proposal);
    
      // Compute the sum of two vectors in sequential for validation.
    for (size_t i = 0; i < sum_sequential.size(); i++) {
      sum_sequential.at(i) = a.at(i) + b.at(i);
      sum_sequential.at(i) = 5*sum_sequential.at(i);
    }

  // Verify that the two vectors are equal.  
    for (size_t i = 0; i < sum_sequential.size(); i++) {
       if (sum_syclParallel.at(i) != sum_sequential.at(i)) {
          std::cout << "A = " << a.at(i) << "\t" << "B = " << b.at(i) << "\t"
          << "sum sequential = " << sum_sequential.at(i) << "\t"
          << "sum sycl = " << sum_syclParallel.at(i) << "\n";
          std::cout << "Vector add failed on SYCL device.\n";
          return -1;
       }
    }
    std::cout << "Vector add Successful on device.\n";

  // Verify that the two vectors are equal.  
    for (size_t i = 0; i < sum_sequential.size(); i++) {
       if (sum_tbbParallel.at(i) != sum_sequential.at(i)) {
          std::cout << "Vector add failed on TBB.\n";
          return -1;
       }
    }
    std::cout << "Vector add Successful on TBB.\n";

  // Verify that the two vectors are equal.  
    for (size_t i = 0; i < sum_sequential.size(); i++) {
       if (sum_tbbParallel_nw_proposal.at(i) != sum_sequential.at(i)) {
          std::cout << "A = " << a.at(i) << "\t" << "B = " << b.at(i) << "\t"
          << "sum sequential = " << sum_sequential.at(i) << "\t"
          << "sum TBB = " << sum_tbbParallel_nw_proposal.at(i) << "\n";
          std::cout << "Vector add failed on TBB using new proposal.\n";
          return -1;
       }
    }
    std::cout << "Vector add Successful on TBB using Fixed Task arena.\n";
    std::cout << "The VectorAdd_tbb took: " << duration_VectorAdd_tbb.count() << std::endl;
    std::cout << "The VectorAdd_tbb_new_proposal took: " << duration_VectorAdd_tbb_new_proposal.count() << std::endl;
    
}