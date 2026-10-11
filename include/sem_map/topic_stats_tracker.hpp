// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"
#include "rclcpp/rclcpp.hpp"

namespace sem_map {

/**
 * @brief Rolling-window rate and latency stats for one topic, published on `~/<short_name>/hz`.
 */
class TopicStatsTracker {
 public:
  TopicStatsTracker(rclcpp::Node* node, const std::string& topic_name,
                    const std::string& short_name, size_t window_size = 100)
      : node_(node),
        topic_name_(topic_name),
        window_size_(window_size),
        total_count_(0),
        has_new_messages_(false) {
    std::string full_topic_name = "~/" + short_name + "/hz";
    hz_pub_ = node_->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(full_topic_name, 10);
  }

  void record_message(double reception_header_latency_ms) {
    total_count_.fetch_add(1, std::memory_order_relaxed);
    push_time();
    push_latency(reception_header_latency_ms);
    has_new_messages_ = true;
  }

  void publish_stats() {
    if (!has_new_messages_) {
      return;
    }

    double fps = compute_fps();
    double avg_latency = compute_avg_latency();
    uint64_t total = total_count_.load();

    auto msg = diagnostic_msgs::msg::DiagnosticArray();
    msg.header.stamp = node_->get_clock()->now();
    msg.header.frame_id = "";

    auto status = diagnostic_msgs::msg::DiagnosticStatus();
    status.name = topic_name_;
    status.hardware_id = "sem_map_node";
    status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
    status.message = "Topic statistics";

    auto kv_fps = diagnostic_msgs::msg::KeyValue();
    kv_fps.key = "fps";
    kv_fps.value = std::to_string(fps);
    status.values.push_back(kv_fps);

    auto kv_latency = diagnostic_msgs::msg::KeyValue();
    kv_latency.key = "latency_ms";
    kv_latency.value = std::to_string(avg_latency);
    status.values.push_back(kv_latency);

    auto kv_total = diagnostic_msgs::msg::KeyValue();
    kv_total.key = "total_messages";
    kv_total.value = std::to_string(total);
    status.values.push_back(kv_total);

    msg.status.push_back(status);
    hz_pub_->publish(msg);
    has_new_messages_ = false;
  }

  double get_fps() const { return compute_fps(); }
  double get_avg_latency() const { return compute_avg_latency(); }
  uint64_t get_total_count() const { return total_count_.load(); }
  size_t get_window_size() const { return times_.size(); }

 private:
  rclcpp::Node* node_;
  std::string topic_name_;
  size_t window_size_;
  std::atomic<uint64_t> total_count_;
  std::atomic<bool> has_new_messages_;
  std::deque<int64_t> times_;
  std::deque<double> latencies_;
  mutable std::mutex times_mutex_;
  mutable std::mutex latencies_mutex_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr hz_pub_;

  void push_time() {
    std::lock_guard<std::mutex> lk(times_mutex_);
    int64_t now_ns = node_->get_clock()->now().nanoseconds();
    times_.push_back(now_ns);
    while (times_.size() > window_size_) {
      times_.pop_front();
    }
  }

  void push_latency(double latency_ms) {
    std::lock_guard<std::mutex> lk(latencies_mutex_);
    latencies_.push_back(latency_ms);
    while (latencies_.size() > window_size_) {
      latencies_.pop_front();
    }
  }

  double compute_fps() const {
    std::lock_guard<std::mutex> lk(times_mutex_);
    if (times_.size() < 2) {
      return 0.0;
    }
    int64_t dt_ns = times_.back() - times_.front();
    if (dt_ns <= 0) {
      return 0.0;
    }
    double dt_s = static_cast<double>(dt_ns) / 1e9;
    return static_cast<double>(times_.size() - 1) / dt_s;
  }

  double compute_avg_latency() const {
    std::lock_guard<std::mutex> lk(latencies_mutex_);
    if (latencies_.empty()) {
      return 0.0;
    }
    double sum = 0.0;
    for (double latency : latencies_) {
      sum += latency;
    }
    return sum / latencies_.size();
  }
};

}  // namespace sem_map
