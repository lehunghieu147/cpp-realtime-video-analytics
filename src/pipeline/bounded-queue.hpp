#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

// Thread-safe bounded queue with drop-oldest overflow policy.
// - push(): if full, drops oldest item to make room (never blocks producer)
// - pop(): blocks until item available or stop() is called
// - stop(): unblocks all waiting consumers, returns nullopt
template <typename T>
class BoundedQueue {
public:
  explicit BoundedQueue(size_t maxSize) : maxSize_(maxSize) {}

  // Push item into queue. If full, drops oldest (front) to make room.
  void push(T item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.size() >= maxSize_) {
        queue_.pop();  // drop oldest
      }
      queue_.push(std::move(item));
    }
    notEmpty_.notify_one();
  }

  // Pop item from queue. Blocks until available or stopped.
  // Returns nullopt if queue is stopped and empty.
  std::optional<T> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    notEmpty_.wait(lock, [this] { return !queue_.empty() || stopped_; });

    if (queue_.empty()) return std::nullopt;

    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  // Pop with timeout. Returns nullopt if timeout expires, queue is empty, or stopped.
  // Useful for main loops that need to check external flags (e.g., SIGINT).
  template <typename Duration>
  std::optional<T> tryPopFor(Duration timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    bool ready = notEmpty_.wait_for(lock, timeout,
                                     [this] { return !queue_.empty() || stopped_; });

    if (!ready || queue_.empty()) return std::nullopt;

    T item = std::move(queue_.front());
    queue_.pop();
    return item;
  }

  // Signal all waiters to stop. After this, pop() returns nullopt once drained.
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopped_ = true;
    }
    notEmpty_.notify_all();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

  bool isStopped() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stopped_;
  }

private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable notEmpty_;
  size_t maxSize_;
  bool stopped_ = false;
};
