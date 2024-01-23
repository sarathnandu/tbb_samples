#include <iostream>
#include <functional>
#include <stack>
#include <exception>
#include <sstream>
#include <array>

#include "my_utils.hpp"
#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/task.h"


class myexception : public std::exception
{
    virtual const char* what() const throw()
    {
        return "My exception : Switch on GPU";
    }
} myex;

static volatile bool gpu_enabled = false;
constexpr float ratio = 0.5; // CPU or GPU offload ratio
constexpr float alpha = 0.5; // coeff for triad calculation

constexpr std::size_t array_size = 16;
std::array<float, array_size> a_array; // input array
std::array<float, array_size> b_array; // input array
std::array<float, array_size> c_array; // output array

class AsyncActivity {
    float offload_ratio;
    std::atomic<bool> submit_flag;
    tbb::task::suspend_point suspend_point;
    std::thread service_thread;

public:
    AsyncActivity() : offload_ratio(0), submit_flag(false),
        service_thread([this] {
        // Wait until the job will be submitted into the async activity
        while (!submit_flag)
            std::this_thread::yield();

        std::size_t array_size_sycl = std::ceil(array_size * offload_ratio);

        // Note that this lambda will be executed concurrently with the task
        // passed into tbb::task_group
        std::stringstream sstream;
        std::cout << "The thread " << GetCurrentThreadId() << " " << "execute async \n";
        sstream << "start index for GPU = 0; end index for GPU = "
            << array_size_sycl << std::endl;
        std::cout << sstream.str();
        const float coeff = alpha; // coeff is a local variable
        if (!gpu_enabled)
           throw myex;

        { // starting SYCL code
            sycl::range<1> n_items{array_size_sycl};
            sycl::buffer<cl_float, 1> a_buffer(a_array.data(), n_items);
            sycl::buffer<cl_float, 1> b_buffer(b_array.data(), n_items);
            sycl::buffer<cl_float, 1> c_buffer(c_array.data(), n_items);

            sycl::queue q;
            q.submit([&](sycl::handler& h) {
                auto a_accessor = a_buffer.get_access<sycl::access::mode::read>(h);
                auto b_accessor = b_buffer.get_access<sycl::access::mode::read>(h);
                auto c_accessor = c_buffer.get_access<sycl::access::mode::write>(h);

                h.parallel_for(n_items, [=](sycl::id<1> index) {
                    c_accessor[index] = a_accessor[index] + coeff * b_accessor[index];
                    }); // end of the kernel
                }).wait();
        }

        // Pass a signal into the main thread that the GPU work is completed
        tbb::task::resume(suspend_point);
            }) {}

    ~AsyncActivity() {
        service_thread.join();
    }
    void submit(float ratio, tbb::task::suspend_point sus_point) {
        offload_ratio = ratio;
        suspend_point = sus_point;
        submit_flag = true;
    }
}; // class AsyncActivity

void test_resumable_task() {
    oneapi::tbb::task_arena arna;
    oneapi::tbb::task_group tg;
    static int array_cnt[5];
    const int N = 1000;
    const size_t work_size = 5;
    static int task_cnt = 0;
    AsyncActivity activity;
    try {
        std::cout << "The thread " << GetCurrentThreadId() << " " << "initiate async \n";
        tbb::task::suspend([&](tbb::task::suspend_point suspend_point) {
            activity.submit(ratio, suspend_point);
            });
        tg.wait();
    }
    catch (std::exception& e) {
        std::cout << "The current task count : " << task_cnt << std::endl;
        std::cout << e.what() << '\n';
        return;
    }
}

int main(int argc, char* argv[]) {
    oneapi::tbb::task_scheduler_handle handle;
    auto default_num_threads = get_default_num_threads();
    handle = oneapi::tbb::task_scheduler_handle{ oneapi::tbb::attach{} };

    test_resumable_task();

    double sum = 0;
    try {
        oneapi::tbb::finalize(handle);
        // oneTBB worker threads are terminated at this point.
    }
    catch (const oneapi::tbb::unsafe_wait&) {
        std::cerr << "Failed to terminate the worker threads." << std::endl;
    }
    return 0;
}