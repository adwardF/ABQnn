#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

#include "umat_auxlib.h"

int main(int argc, char *argv[])
{
    const char *model_path = "test_NH_3D.pt";
    if (argc > 1)
    {
        model_path = argv[1];
    }

    constexpr int kThreadCount = 8;
    constexpr int kCallsPerThread = 12;

    std::cout << "ABQnn Parallel IPC Test" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Threads: " << kThreadCount << ", calls/thread: " << kCallsPerThread << std::endl;

    std::atomic<int> total_failures{0};
    std::atomic<int> completed_calls{0};
    std::atomic<int> finished_threads{0};
    std::mutex error_mutex;
    std::vector<std::string> error_messages;
    std::mutex done_mutex;
    std::condition_variable done_cv;

    auto worker = [&](int thread_id)
    {
        for (int call_id = 0; call_id < kCallsPerThread; ++call_id)
        {
            double F[3][3] = {
                {1.0 + 0.01 * (thread_id + 1), 0.001 * (call_id + 1), 0.0},
                {0.0, 1.0 + 0.005 * (call_id + 1), 0.0},
                {0.0, 0.0, 1.0}};

            const double det_scale = F[0][0] * F[1][1];
            F[2][2] = 1.0 / det_scale;

            double mat_par[2] = {1.0, 10.0};
            double psi = 0.0;
            double cauchy6[6] = {0.0};
            double ddsdde[6][6] = {{0.0}};

            int err = invoke_pt(model_path, &F[0][0], mat_par, 2, &psi, cauchy6, &ddsdde[0][0]);
            if (err != 0)
            {
                total_failures.fetch_add(1);
                std::ostringstream oss;
                oss << "thread " << thread_id << ", call " << call_id << ": invoke_pt error=" << err;
                std::lock_guard<std::mutex> lock(error_mutex);
                error_messages.push_back(oss.str());
                continue;
            }

            if (!std::isfinite(psi))
            {
                total_failures.fetch_add(1);
                std::ostringstream oss;
                oss << "thread " << thread_id << ", call " << call_id << ": non-finite psi";
                std::lock_guard<std::mutex> lock(error_mutex);
                error_messages.push_back(oss.str());
                continue;
            }

            for (double v : cauchy6)
            {
                if (!std::isfinite(v))
                {
                    total_failures.fetch_add(1);
                    std::ostringstream oss;
                    oss << "thread " << thread_id << ", call " << call_id << ": non-finite Cauchy value";
                    std::lock_guard<std::mutex> lock(error_mutex);
                    error_messages.push_back(oss.str());
                    break;
                }
            }

            completed_calls.fetch_add(1);
        }

        finished_threads.fetch_add(1);
        done_cv.notify_one();
    };

    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i)
    {
        threads.emplace_back(worker, i);
    }

    for (auto &t : threads)
    {
        t.detach();
    }

    {
        std::unique_lock<std::mutex> lock(done_mutex);
        const bool all_finished = done_cv.wait_for(
            lock,
            std::chrono::seconds(60),
            [&]() { return finished_threads.load() == kThreadCount; });

        if (!all_finished)
        {
            const int done = completed_calls.load();
            std::cout << "Parallel IPC test timed out: completed " << done
                      << " / " << (kThreadCount * kCallsPerThread)
                      << " calls within 60s." << std::endl;
            return 1;
        }
    }

    const int failures = total_failures.load();
    if (failures != 0)
    {
        std::cout << "Parallel IPC test failed with " << failures << " failing calls." << std::endl;
        for (const auto &msg : error_messages)
        {
            std::cout << "  - " << msg << std::endl;
        }
        return 1;
    }

    std::cout << "Parallel IPC test passed: " << (kThreadCount * kCallsPerThread)
              << " requests completed successfully." << std::endl;
    return 0;
}
