#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <scheduler_common.h>

static const int numThreads = 10;
static const int yieldThreshold = 100;
static int count = 0;
std::mutex mtx;
std::condition_variable cv;
void test_yield();
void test_pause();

int main() {

   test_yield();
   return 0;
}

// Function to be executed by the threads
void threadFunction_arena1(int id) {
    std::unique_lock<std::mutex> lock(mtx);
    count++;
    if (count == numThreads) {
        cv.notify_all(); // Notify all waiting threads that barrier is reached
    }
    
    // Wait for all threads to reach the barrier
    cv.wait(lock, []{ return count == numThreads; });
    for (unsigned int i = 0; i < yieldThreshold; i++) {
       tbb::detail::d0::yield();
    }
    //std::cout << "Thread " << id << " passed the barrier." << std::endl;
}

void test_pause() {
   auto start_prol_pause = std::chrono::steady_clock::now();
   for (unsigned int i = 0; i<100; i++) {
      tbb::detail::r1::prolonged_pause_impl();
    }
   auto end_prol_pause = std::chrono::steady_clock::now();
   std::chrono::duration<double> duration_prol_pause = std::chrono::duration_cast<std::chrono::duration<double>>(end_prol_pause - start_prol_pause);
   std::cout << "The prolonged_pause_impl took: " << duration_prol_pause.count() << std::endl;
}
void test_yield() {
   std::vector<std::thread> threads;
   std::chrono::duration<double> duration_yield;
   auto start_yield =  std::chrono::steady_clock::now();
   for (int i = 0; i < numThreads; ++i) {
      threads.push_back(std::thread(threadFunction_arena1, i));
   }
   for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
   auto end_yield = std::chrono::steady_clock::now();
   duration_yield = std::chrono::duration_cast<std::chrono::duration<double>>(end_yield - start_yield);
   std::cout << "The yield: took: " << duration_yield.count() << " seconds" << std::endl;
}
