#pragma once

#include <cstdint>
#include <thread>
#include <vector>
#include <algorithm>

/**
 * @brief Simple parallel execution helper
 */
template <typename CallableType, typename IntableType>
inline void parallelize(
    const CallableType &fn,
    const IntableType range,
    const int num_workers
) {
    if (num_workers <= 1 || range <= 1) {
        for (IntableType i = 0; i < range; ++i) {
            fn(i);
        }
        return;
    }

    int workers = std::min((int)range, num_workers);
    std::vector<std::thread> threads;
    threads.reserve(workers);

    IntableType task_num_per_worker = range / workers;
    IntableType remainder = range % workers;
    IntableType current_start = 0;

    for (int id = 0; id < workers; ++id) {
        IntableType count = task_num_per_worker + (id < remainder ? 1 : 0);
        IntableType next_start = current_start + count;

        threads.emplace_back([current_start, next_start, &fn]() {
            for (IntableType i = current_start; i < next_start; ++i) {
                fn(i);
            }
        });
        current_start = next_start;
    }

    for (auto &t : threads) {
        t.join();
    }
}

