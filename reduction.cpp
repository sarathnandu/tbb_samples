//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of SYCL. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// •	A one dimensional array of data.
// •	A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/global_control.h"
#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <sstream>
#include <barrier>
#include <syncstream>
#include "oneapi/tbb/parallel_for.h"
#if FPGA_HARDWARE || FPGA_EMULATOR || FPGA_SIMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif


using namespace sycl;

// Cache list of devices
bool cached = false;
std::vector<device> devices;

// num_repetitions: How many times to repeat the kernel invocation
size_t num_repetitions = 1;
// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<int> IntVector;

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


int dotProduct_sycl(queue& q, const IntVector& aVector, const IntVector& bVector) {

    buffer a_buf(aVector);
    buffer b_buf(bVector);


    device dev = sycl::device(sycl::cpu_selector_v);
    auto dot_num_groups = dev.get_info<info::device::max_compute_units>();
    auto dot_wgsize = dev.get_info<info::device::native_vector_width_double>() * 2;

    range<1> num_items{ aVector.size() };
    buffer<int, 1> d_sum(dot_num_groups);
    std::vector<int> pfxSum(aVector.size(), 0);
    buffer pfxSum_buf(pfxSum.data(), num_items);

    q.submit([&](handler& cgh)
        {
            accessor ka(a_buf, cgh, read_only);
            accessor kb(b_buf, cgh, read_only);
            accessor ksum(d_sum, cgh, write_only, no_init);
            accessor debug_pfxsum(pfxSum_buf, cgh, write_only, no_init);

            auto wg_sum = accessor<int, 1, access::mode::read_write, access::target::local>(range<1>(dot_wgsize), cgh);

            size_t N = aVector.size();
            cgh.parallel_for<class dot_kernel_2>(nd_range<1>(dot_num_groups * dot_wgsize, dot_wgsize), [=](nd_item<1> item)
                {
                    size_t i = item.get_global_id(0);
                    size_t li = item.get_local_id(0);
                    size_t global_size = item.get_global_range()[0];

                    wg_sum[li] = {};
                    for (; i < N; i += global_size) {
                        wg_sum[li] += ka[i] * kb[i];
                        //debug_pfxsum[i] = ka[i] + kb[i];
                    }

                    size_t local_size = item.get_local_range()[0];
                    for (int offset = local_size / 2; offset > 0; offset /= 2)
                    {
                        item.barrier(sycl::access::fence_space::local_space);
                        if (li < offset)
                            wg_sum[li] += wg_sum[li + offset];
                    }

                    if (li == 0)
                        ksum[item.get_group(0)] = wg_sum[0];
                });
        });
    q.wait();
    int sum{};
    auto h_sum = d_sum.get_access<access::mode::read>();
    for (int i = 0; i < dot_num_groups; i++)
    {
        //std::cout << h_sum[i] << "\t";
        sum += h_sum[i];
    }
    return sum;
}

int dotProduct_tbb(const std::vector<int>& aVector, const std::vector<int>& bVector) {
    device dev = sycl::device(sycl::cpu_selector_v);
    int noOfWrkGrps = dev.get_info<info::device::max_compute_units>();
    int wrkGrpSize = dev.get_info<info::device::native_vector_width_double>() * 2;

    oneapi::tbb::global_control c(tbb::global_control::max_allowed_parallelism, noOfWrkGrps);
    oneapi::tbb::task_arena tbb_arena(noOfWrkGrps);

    std::barrier sync(noOfWrkGrps);
    std::mutex mtx;
    int reductionArr[12] = { 0 };
    int sum = 0;

    tbb_arena.execute([&] {
        oneapi::tbb::parallel_for(tbb::blocked_range<int>(0, noOfWrkGrps),
        [&](tbb::blocked_range<int> r)
            {
                int local_result = 0;
                for (int j = (r.begin()* wrkGrpSize); j < (r.end() * wrkGrpSize); ++j)
                {
                    local_result += aVector[j] * bVector[j];
                }
                int thrIndex = tbb::this_task_arena::current_thread_index();
                reductionArr[thrIndex] = local_result;
                sync.arrive_and_wait();
                for (int i = noOfWrkGrps / 2; i > 0; i >>= 1) {
                    if (thrIndex < i) {
                        reductionArr[thrIndex] = reductionArr[thrIndex] + reductionArr[thrIndex + i];
                    }
                    // handle unpaired index for odd values
                    if ((i % 2 == 1) && (thrIndex == (i -1) )) {
                        reductionArr[thrIndex] = reductionArr[thrIndex] + reductionArr[thrIndex + 1];
                    }

                    sync.arrive_and_wait();
                }
            }, tbb::static_partitioner());
        });
    sum = sum + reductionArr[0];

    return sum;
}

int dotProduct_tbbCPU(const std::vector<int>& aVector, const std::vector<int>& bVector) {
    oneapi::tbb::global_control c(tbb::global_control::max_allowed_parallelism, 64);

    int noOfWrkGrps = 12;
    int wrkGrpSize = 8;
    const int thrCount = 2;
    oneapi::tbb::task_arena arena(thrCount);
    std::barrier sync(thrCount);
    std::vector<int> wgArr(noOfWrkGrps, 0);
    for (int idx = 0; idx < noOfWrkGrps; idx++) {
        std::vector<int> reductionArr(thrCount, 0);
        arena.execute([&] {
            oneapi::tbb::parallel_for(tbb::blocked_range<int>(0, wrkGrpSize),
            [&](tbb::blocked_range<int> r)
                {
                    int local_result = 0;
                    int offset = idx * wrkGrpSize;
                    for (int j = r.begin(); j < r.end(); ++j)
                    {
                        local_result += aVector[j + offset] * bVector[j + offset];
                    }
                    int thrIndex = tbb::this_task_arena::current_thread_index();

                    reductionArr[thrIndex] = local_result;
                    sync.arrive_and_wait();
                    // only 2 partitions; thread 0 does the reduction
                    if (0 == thrIndex) { wgArr[idx] += reductionArr[0] + reductionArr[1]; }
                }, tbb::static_partitioner());
            });
    }
    int sum{};
    for (int i = 0; i < noOfWrkGrps; i++) {
        sum = sum + wgArr[i];
    }
    return sum;
}

int dotProduct_seq(const std::vector<int>& aVector, const std::vector<int>& bVector) {
    int result = 0;
    for (int i = 0; i < aVector.size(); i++) {
        result = result + (aVector[i] * bVector[i]);
    }
    return result;
}

int tbbNestedImpl(const std::vector<int>& aVector, const std::vector<int>& bVector, int wg) {
    oneapi::tbb::global_control c(tbb::global_control::max_allowed_parallelism, 64);

    int wrkGrpSize = 8;
    const int thrCount = 2;
    int sum = 0;
    tbb::task_arena sgArena(thrCount, 1);
    std::barrier sync(thrCount);
    std::vector<int> reductionArr(thrCount, 0);
    sgArena.execute([&] {
        oneapi::tbb::parallel_for(tbb::blocked_range<int>(0, wrkGrpSize),
        [&](tbb::blocked_range<int> r)
            {
                int local_result = 0;
                int offset = wg * wrkGrpSize;
                for (int j = r.begin(); j < r.end(); ++j)
                {
                    local_result += aVector[j + offset] * bVector[j + offset];
                }
                int thrIndex = tbb::this_task_arena::current_thread_index();

                reductionArr[thrIndex] = local_result;
                sync.arrive_and_wait();
                // only 2 partitions; thread 0 does the reduction
                if (0 == thrIndex) { sum = reductionArr[0] + reductionArr[1]; }
            }, tbb::static_partitioner());
        });
    return sum;
}

int dotProduct_tbbGpuEmulation(const std::vector<int>& aVector, const std::vector<int>& bVector) {
    device dev = sycl::device(sycl::cpu_selector_v);
    int noOfWrkGrps = dev.get_info<info::device::max_compute_units>();
    //int wrkGrpSize = dev.get_info<info::device::native_vector_width_double>() * 2;

    oneapi::tbb::global_control c(tbb::global_control::max_allowed_parallelism, 64);
    std::barrier sync(noOfWrkGrps);

    // There are total 12 Workgroups each executed in a single fixed work group arena
    tbb::task_arena wgArena(noOfWrkGrps, 1);
    std::vector<int> reductionArr(noOfWrkGrps, 0);
    int sum = 0;

    wgArena.execute([&] {
        oneapi::tbb::parallel_for(tbb::blocked_range<int>(0, noOfWrkGrps),
        [&](tbb::blocked_range<int> r)
            {
                int thrIndex = tbb::this_task_arena::current_thread_index();
                reductionArr[thrIndex] = tbbNestedImpl(aVector, bVector, r.begin());
                sync.arrive_and_wait();
                for (int i = noOfWrkGrps / 2; i > 0; i >>= 1) {
                    if (thrIndex < i) {
                        reductionArr[thrIndex] = reductionArr[thrIndex] + reductionArr[thrIndex + i];
                    }
                    // handle unpaired index for odd values
                    if ((i % 2 == 1) && (thrIndex == (i - 1))) {
                        reductionArr[thrIndex] = reductionArr[thrIndex] + reductionArr[thrIndex + 1];
                    }

                    sync.arrive_and_wait();
                }
            }, tbb::static_partitioner());
        });
    sum = sum + reductionArr[0];

    return sum;
}


//************************************
// Initialize the vector from 0 to vector_size - 1
//************************************
void InitializeVector(IntVector &a) {
    for (size_t i = 0; i < a.size(); i++) a.at(i) = i;
}

//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
    // Change num_repetitions if it was passed as argument
    if (argc > 2) num_repetitions = std::stoi(argv[2]);
    // Change vector_size if it was passed as argument
    if (argc > 1) vector_size = std::stoi(argv[1]);
    // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // Intel extension: FPGA emulator selector on systems without FPGA card.
  auto selector = sycl::ext::intel::fpga_emulator_selector_v;
#elif FPGA_SIMULATOR
  // Intel extension: FPGA simulator selector on systems without FPGA card.
  auto selector = sycl::ext::intel::fpga_simulator_selector_v;
#elif FPGA_HARDWARE
  // Intel extension: FPGA selector on systems with FPGA card.
  auto selector = sycl::ext::intel::fpga_selector_v;
#else
  // The default device selector will select the most performant device.
  auto selector = cpu_selector_v;
#endif

    // Create vector objects with "vector_size" to store the input and output data.
    IntVector a, b, sum_sequential, sum_parallel;

    std::vector<int> in_a(96, 1);
    std::vector<int> in_b(96, 1);
    std::vector<int> op_c(96, 0);
    int dot_sycl = 0;
    // Vector dotproduct in SYCL
    try {
        queue q(selector, exception_handler);

        // Print out the device information used for the kernel code.
        std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>() << "\n";
        std::cout << "Vector size: " << in_a.size() << "\n";

        // Vector dot product in SYCL
        dot_sycl = dotProduct_sycl(q, in_a, in_b);
    }
    catch (exception const& e) {
        std::cout << "An exception is caught for vector add.\n";
        std::terminate();
    }

    std::cout << "The dot product SYCL is : " << dot_sycl << std::endl;

    auto dotTbb = dotProduct_tbb(in_a, in_b);
    std::cout << "The dot product TBB with vectorization : " << dotTbb << std::endl;

    auto dotCPU = dotProduct_tbbCPU(in_a, in_b);
    std::cout << "The dot product TBB sub group parallel_for is : " << dotCPU << std::endl;

    auto dotGpuEmul = dotProduct_tbbGpuEmulation(in_a, in_b);
    std::cout << "The dot product TBB parallel_for gpu emulation is : " << dotGpuEmul << std::endl;

    auto dotSeq = dotProduct_seq(in_a, in_b);
    std::cout << "The dot product Sequential is : " << dotSeq << std::endl;
    return 0;
}