// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <deque>
#include <mutex>
#include <vector>

namespace sem_map {

/**
 * @brief Thread-safe fixed-capacity buffer; pushing when full drops the oldest entry.
 * @tparam T Element type
 */
template <typename T>
class RingBuffer {
 public:
  explicit RingBuffer(size_t capacity) : capacity_(capacity) {}

  void push(const T& item) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (buffer_.size() == capacity_) {
      buffer_.pop_front();
    }
    buffer_.push_back(item);
  }

  void push(T&& item) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (buffer_.size() == capacity_) {
      buffer_.pop_front();
    }
    buffer_.push_back(std::move(item));
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return buffer_.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return buffer_.size();
  }

  const T& back() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return buffer_.back();
  }

  const T& front() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return buffer_.front();
  }

  void pop_front() {
    std::lock_guard<std::mutex> lk(mutex_);
    buffer_.pop_front();
  }

  void clear() {
    std::lock_guard<std::mutex> lk(mutex_);
    buffer_.clear();
  }

  // Atomically move-out the front element and remove it. Returns false if empty.
  // Safe alternative to front() + pop_front() which has a race between unlock and copy.
  bool try_pop_front(T& out) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (buffer_.empty()) return false;
    out = std::move(buffer_.front());
    buffer_.pop_front();
    return true;
  }

  // Copy the back element under the lock. Returns false if empty.
  bool back_copy(T& out) const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (buffer_.empty()) return false;
    out = buffer_.back();
    return true;
  }

  // Thread-safe iteration with a callback
  template <typename Func>
  void for_each(Func func) const {
    std::lock_guard<std::mutex> lk(mutex_);
    for (const auto& item : buffer_) {
      func(item);
    }
  }

  // Binary search for two elements bracketing a target key.
  // Requires: buffer is sorted ascending by key_func(item).
  // Returns true if bracket found, filling 'before' and 'after' by copy (safe after unlock).
  // Also fills oldest_key/newest_key for diagnostics. buffer_size is set to current size.
  template <typename KeyFunc, typename KeyType>
  bool find_bracket(KeyFunc key_func, KeyType target, T& before, T& after, KeyType& oldest_key,
                    KeyType& newest_key, size_t& buffer_size) const {
    std::lock_guard<std::mutex> lk(mutex_);
    buffer_size = buffer_.size();
    if (buffer_size < 2) return false;

    oldest_key = key_func(buffer_.front());
    newest_key = key_func(buffer_.back());

    // Target outside buffer range
    if (target < oldest_key || target > newest_key) return false;

    // Binary search: find first index where key >= target
    size_t lo = 0, hi = buffer_size;
    while (lo < hi) {
      size_t mid = lo + (hi - lo) / 2;
      if (key_func(buffer_[mid]) < target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    // lo = first element with key >= target
    if (lo > 0) {
      before = buffer_[lo - 1];  // copy under lock
      after = buffer_[lo];       // copy under lock
    } else {
      // target == oldest key exactly
      before = buffer_[0];
      after = buffer_[0];
    }
    return true;
  }

 private:
  size_t capacity_;
  mutable std::mutex mutex_;
  std::deque<T> buffer_;
};

}  // namespace sem_map
