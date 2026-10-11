// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include "sem_map/ring_buffer.hpp"
#include "sem_map/topic_stats_tracker.hpp"

namespace sem_map {

/**
 * @brief Per-topic point cloud state: buffer, cached base-to-sensor transform and statistics.
 */
class PointCloudSensor {
 public:
  PointCloudSensor(rclcpp::Node* node, const std::string& name, const std::string& pointcloud_topic,
                   size_t buffer_size)
      : name_(name),
        pointcloud_topic_(pointcloud_topic),
        buffer_(buffer_size),
        base_T_pointcloud_(),
        have_base_T_pointcloud_(false) {
    // Create statistics tracker
    pc_stats_ = std::make_unique<TopicStatsTracker>(node, pointcloud_topic_, "pointcloud/" + name_);
  }

  // Getters
  const std::string& get_name() const { return name_; }
  const std::string& get_pointcloud_topic() const { return pointcloud_topic_; }

  RingBuffer<sensor_msgs::msg::PointCloud2>& get_buffer() { return buffer_; }
  const RingBuffer<sensor_msgs::msg::PointCloud2>& get_buffer() const { return buffer_; }

  bool has_base_T_pointcloud() const { return have_base_T_pointcloud_.load(); }
  const geometry_msgs::msg::TransformStamped& get_base_T_pointcloud() const {
    return base_T_pointcloud_;
  }
  void set_base_T_pointcloud(const geometry_msgs::msg::TransformStamped& transform) {
    base_T_pointcloud_ = transform;
    have_base_T_pointcloud_ = true;
  }

  TopicStatsTracker* get_pc_stats() { return pc_stats_.get(); }

  // Subscription (public for node to access)
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;

 private:
  std::string name_;
  std::string pointcloud_topic_;

  RingBuffer<sensor_msgs::msg::PointCloud2> buffer_;

  geometry_msgs::msg::TransformStamped base_T_pointcloud_;
  std::atomic<bool> have_base_T_pointcloud_;

  std::unique_ptr<TopicStatsTracker> pc_stats_;
};

}  // namespace sem_map
