#include <iostream>
#include <barrier>
#include <syncstream>
#include <set>
#include <vector>
#include "my_utils.hpp"

#include "oneapi/tbb/task_arena.h"
#include "oneapi/tbb/parallel_for.h"


void nested_arena_par_for() {
    int elements = 4800;
    std::vector<double> A(elements);
    std::barrier sync_point(4 /*No of partitions*/);
    my_barrier nested_barrier(2);
    oneapi::tbb::task_group_context context[2];
    oneapi::tbb::task_arena arenas[2];
    oneapi::tbb::task_group groups[2];
    oneapi::tbb::task_group_context contexts[2];
    oneapi::tbb::task_arena::constraints arena_constraints[2];
    std::set<int> arena_unq_threads[2];
    pinning_observer arena1_observer(arenas[0]);

    tbb::blocked_range<size_t> range(0, elements, (elements / 3));
    auto tbb_partitioner = tbb::simple_partitioner{};
    arenas[0].execute([&] {
        tbb::parallel_for(tbb::blocked_range<int>(0, elements, (elements / 3)), [&](tbb::blocked_range<int> r) {
            for (auto i = r.begin(); i < r.end(); i++) {
                A[i] = 0.1;
                A[i] = sin(A[i] / 0.7);
            }
            //const std::lock_guard<std::mutex> lock(arena_thr_mutex);
            auto thread_id = GetCurrentThreadId();
            auto worker_id = tbb::this_task_arena::current_thread_index();
            std::string tbb_lambda = " Thread: " + std::to_string(thread_id) +
                " Worker_ID: " + std::to_string(worker_id) +
                " executing Range: " + std::to_string(r.begin()) +
                " to " + std::to_string(r.end()) + "\n";
            std::osyncstream(std::cout) << tbb_lambda;

            if (!arena_unq_threads[0].contains(thread_id)) {
#if 0
                arenas[1].execute([&] {
                    groups[0].defer([&] {
                        nested_barrier.nested_task1();
                        });
                    groups[1].run([&] {
                        nested_barrier.nested_task1();
                        nested_barrier.nested_task2();
                        groups[1].wait();
                        });
                    }
                );
#endif
                sync_point.arrive_and_wait();
                //barrier.wait();
            }
            arena_unq_threads[0].insert(thread_id);

            }, tbb_partitioner);
        });
}
