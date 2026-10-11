// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <cstddef>
#include <deque>
#include <mutex>

namespace sem_map {

/**
 * @brief Thread-safe rolling window statistics for execution time tracking
 *
 * Tracks execution times over a fixed window size and computes average.
 * Used to measure active processing time (excluding idle/sleep periods).
 */
class ThreadStats {
 public:
  /**
   * @brief Construct a ThreadStats tracker
   * @param window_size Number of samples to keep in rolling window (default: 100)
   */
  explicit ThreadStats(size_t window_size = 100) : window_size_(window_size) {}

  /**
   * @brief Record a new execution time measurement
   * @param time_ms Execution time in milliseconds
   */
  void record(double time_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    times_ms_.push_back(time_ms);
    while (times_ms_.size() > window_size_) {
      times_ms_.pop_front();
    }
  }

  /**
   * @brief Get average execution time over the window
   * @return Average time in milliseconds, or 0.0 if no samples
   */
  double get_avg_ms() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (times_ms_.empty()) {
      return 0.0;
    }
    double sum = 0.0;
    for (double t : times_ms_) {
      sum += t;
    }
    return sum / static_cast<double>(times_ms_.size());
  }

  /**
   * @brief Get number of samples in the window
   * @return Sample count
   */
  size_t get_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return times_ms_.size();
  }

 private:
  std::deque<double> times_ms_;
  mutable std::mutex mutex_;
  size_t window_size_;
};

}  // namespace sem_map
