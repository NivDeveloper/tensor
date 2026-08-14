#pragma once

// The process-wide worker pool and the chunked parallel_for eval runs on;
// configured only through use_threads (Tensor.h).

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <latch>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tensor::detail {

// A nested dispatch on a worker must run serially or it deadlocks.
inline thread_local bool pool_worker = false;

class ThreadPool {
  public:
    explicit ThreadPool(size_t workers) {
        workers_.reserve(workers);
        for (size_t i = 0; i < workers; ++i)
            workers_.emplace_back([this] {
                pool_worker = true;
                run();
            });
    }

    ~ThreadPool() {
        {
            std::lock_guard lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto &w : workers_)
            w.join();
    }

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    size_t workers() const { return workers_.size(); }

    // Fire-and-forget; parallel_for counts completion on a latch.
    void enqueue(std::move_only_function<void()> job) {
        {
            std::lock_guard lk(m_);
            jobs_.push(std::move(job));
        }
        cv_.notify_one();
    }

  private:
    void run() {
        for (;;) {
            std::move_only_function<void()> job;
            {
                std::unique_lock lk(m_);
                cv_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
                if (stop_ && jobs_.empty())
                    return; // drain before stopping
                job = std::move(jobs_.front());
                jobs_.pop();
            }
            job();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::move_only_function<void()>> jobs_;
    std::mutex m_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// A function-local static settles initialization order; the unique_ptr
// joins on replacement and at exit.
inline std::unique_ptr<ThreadPool> &pool_slot() {
    static std::unique_ptr<ThreadPool> p;
    return p;
}

// Minimum elements per chunk, placed by measurement (bench/README.md):
// a dispatch costs ~15-25 us, so small evals must never pay one.
inline constexpr size_t eval_grain = 32768;

// f(begin, end) over disjoint chunks of [0, n), serial below two grains.
// The caller takes chunk 0 and waits, keeping the captures valid. A caller
// whose units are coarser than elements (rows) scales `grain` accordingly.
template <typename F>
void parallel_for(size_t n, F &&f, size_t grain = eval_grain) {
    ThreadPool *p = pool_slot().get();
    if (!p || pool_worker || n < 2 * grain) {
        f(size_t{0}, n);
        return;
    }
    const size_t chunks =
        std::min(p->workers() + 1, n / grain); // every chunk >= grain
    const size_t chunk = (n + chunks - 1) / chunks;
    const size_t spawned = (n + chunk - 1) / chunk - 1; // caller takes chunk 0
    std::latch done(static_cast<std::ptrdiff_t>(spawned));
    for (size_t start = chunk; start < n; start += chunk)
        p->enqueue([&f, &done, start, end = std::min(start + chunk, n)] {
            f(start, end);
            done.count_down();
        });
    f(size_t{0}, chunk);
    done.wait();
}

} // namespace tensor::detail
