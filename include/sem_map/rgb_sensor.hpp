// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "sem_map/ring_buffer.hpp"
#include "sem_map/topic_stats_tracker.hpp"

namespace sem_map {

/**
 * @brief Per-camera state: image buffer, CameraInfo, cached base-to-camera transform, stats.
 */
class RgbSensor {
 public:
  RgbSensor(rclcpp::Node* node, const std::string& name, const std::string& rgb_topic,
            const std::string& info_topic, size_t buffer_size)
      : name_(name),
        rgb_topic_(rgb_topic),
        info_topic_(info_topic),
        buffer_(buffer_size),
        cam_info_(),
        have_cam_info_(false),
        base_T_rgb_(),
        have_base_T_rgb_(false) {
    // Create statistics trackers
    rgb_stats_ = std::make_unique<TopicStatsTracker>(node, rgb_topic_, "rgb/" + name_);
  }

  // Getters
  const std::string& get_name() const { return name_; }
  const std::string& get_rgb_topic() const { return rgb_topic_; }
  const std::string& get_info_topic() const { return info_topic_; }

  RingBuffer<sensor_msgs::msg::Image>& get_buffer() { return buffer_; }
  const RingBuffer<sensor_msgs::msg::Image>& get_buffer() const { return buffer_; }

  sensor_msgs::msg::CameraInfo& get_cam_info() { return cam_info_; }
  const sensor_msgs::msg::CameraInfo& get_cam_info() const { return cam_info_; }

  bool has_cam_info() const { return have_cam_info_.load(); }
  void set_cam_info(const sensor_msgs::msg::CameraInfo& info) {
    cam_info_ = info;
    have_cam_info_ = true;
  }

  bool has_base_T_rgb() const { return have_base_T_rgb_.load(); }
  const geometry_msgs::msg::TransformStamped& get_base_T_rgb() const { return base_T_rgb_; }
  void set_base_T_rgb(const geometry_msgs::msg::TransformStamped& transform) {
    base_T_rgb_ = transform;
    have_base_T_rgb_ = true;
  }

  TopicStatsTracker* get_rgb_stats() { return rgb_stats_.get(); }

  // Subscriptions (public for node to access)
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;

 private:
  std::string name_;
  std::string rgb_topic_;
  std::string info_topic_;

  RingBuffer<sensor_msgs::msg::Image> buffer_;
  sensor_msgs::msg::CameraInfo cam_info_;
  std::atomic<bool> have_cam_info_;

  geometry_msgs::msg::TransformStamped base_T_rgb_;
  std::atomic<bool> have_base_T_rgb_;

  std::unique_ptr<TopicStatsTracker> rgb_stats_;
};

}  // namespace sem_map
