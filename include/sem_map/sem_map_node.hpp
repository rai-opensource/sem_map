// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <sophus/se3.hpp>
#include "grid_map_msgs/msg/grid_map.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "sem_map/buffer_constants.hpp"
#include "sem_map/pointcloud_sensor.hpp"
#include "sem_map/rgb_sensor.hpp"
#include "sem_map/ring_buffer.hpp"
#include "sem_map/sem_map_bindings.hpp"
#include "sem_map/thread_stats.hpp"
#include "sem_map/topic_stats_tracker.hpp"

namespace sem_map {

constexpr int MESSAGE_THROTTLE_TIME_MS = 20000;  // 20 seconds
constexpr double TIME_JUMP_THRESHOLD_SEC =
    5.0;  // Backward jump threshold for rosbag loop detection
constexpr double MAX_PENDING_HOLD_SEC =
    5.0;  // Drop a pending frame stuck waiting for poses longer than this

/**
 * @brief Result of interpolating a pose at a requested timestamp.
 */
enum class PoseInterpolationStatus {
  SUCCESS,            // Successfully interpolated pose
  TOO_OLD,            // Target timestamp is older than oldest pose in buffer
  TOO_NEW,            // Target timestamp is newer than newest pose in buffer
  INSUFFICIENT_POSES  // Not enough poses in buffer (< 2)
};

/**
 * @brief ROS 2 node building a semantic traversability map from camera and lidar input.
 *
 * Buffers incoming messages and runs inference, elevation mapping and publishing on
 * separate threads, so no subscription callback blocks on the pipeline.
 */
class SemMapNode : public rclcpp::Node {
 public:
  explicit SemMapNode(const rclcpp::NodeOptions& options);
  ~SemMapNode();

 private:
  // Shutdown helper (called from pre-shutdown callback)
  void stopThreads();
  template <typename MessageType>
  void process_message(const MessageType& msg, const rclcpp::MessageInfo& info,
                       TopicStatsTracker& stats_tracker, const std::string& msg_type);

  // Single-stream callbacks
  void on_rgb_compressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
  void on_rgb_raw(const sensor_msgs::msg::Image::SharedPtr msg, const rclcpp::MessageInfo& info);
  void on_cam_info_single(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void on_pointcloud_single(const sensor_msgs::msg::PointCloud2::SharedPtr msg,
                            const rclcpp::MessageInfo& info);
  void on_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg,
               const rclcpp::MessageInfo& info);
  void on_pose_odom(const nav_msgs::msg::Odometry::SharedPtr msg, const rclcpp::MessageInfo& info);
  // Shared body of the two pose callbacks.
  void handle_pose(const geometry_msgs::msg::PoseStamped& msg, const rclcpp::MessageInfo& info);

  // (Deprecated) Multi-sensor setup methods (unused in single-stream mode)
  void setup_rgb_sensors();
  void setup_pointcloud_sensors();

  // Processing threads
  void semantic_processing_loop();
  void pointcloud_processing_loop();
  void ros_publishing_loop();

  // Pose interpolation and helpers
  PoseInterpolationStatus interpolate_pose(uint64_t target_stamp_ns,
                                           geometry_msgs::msg::PoseStamped& result);
  bool find_closest_poses(uint64_t target_stamp_ns, geometry_msgs::msg::PoseStamped& before,
                          geometry_msgs::msg::PoseStamped& after);

  // Helper methods
  bool parse_pointcloud_xyz(const sensor_msgs::msg::PointCloud2& cloud, std::vector<float>& points);
  std_msgs::msg::Float32MultiArray create_float32_multiarray(const std::vector<float>& data,
                                                             int rows, int cols);

  // Parameters (aligned with sem_map.yaml)
  std::string pose_topic_;
  std::string pointcloud_topic_;
  std::string image_topic_;
  std::string camera_info_topic_;
  bool image_is_compressed_{true};
  bool pose_is_odometry_{false};
  std::string world_frame_;
  std::string base_frame_;
  std::string pose_frame_;
  size_t rgb_buffer_size_;
  size_t pointcloud_buffer_size_;
  size_t pose_buffer_size_;
  double pointcloud_processing_rate_hz_{10.0};
  double rgb_processing_rate_hz_{10.0};
  double ros_publish_rate_hz_{30.0};
  bool publish_traversability_{true};
  bool publish_segmentation_{true};
  bool publish_confidence_{true};
  bool publish_custom_elevation_{false};  // color_r mode only: publish constant elevation
  float custom_elevation_{0.0f};          // value used when publish_custom_elevation_ is true
  // Map parameters (stored only)
  double map_resolution_{0.1};
  double map_size_{20.0};
  std::string map_backend_{"cupy"};
  std::string map_precision_{"float16"};
  std::string semantic_bev_layer_name_{"trail"};
  // Feature flags
  bool enable_bev_fusion_{true};
  bool enable_height_map_{true};

  // Subscriptions (single-stream)
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr rgb_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr rgb_raw_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
  rclcpp::TimerBase::SharedPtr stats_timer_;

  // Services
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_elevation_map_srv_;

  // Time-jump generation counter: incremented by on_pose when a backward jump
  // is detected. Worker threads snapshot it when they pop a pending_msg and
  // drop the pending if the counter has changed since. A pre-restart frame has
  // a stamp far in the *future* relative to the restarted clock, so comparing
  // stamps against a baseline cannot distinguish it from legitimate new frames;
  // the generation counter can.
  std::atomic<uint64_t> time_jump_counter_{0};

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr traversability_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr segmentation_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr confidence_pub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr grid_map_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr map_center_pub_;

  // TF broadcaster
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  // (Deprecated) Multi-sensor maps
  std::map<std::string, std::unique_ptr<RgbSensor>> rgb_sensors_;
  std::map<std::string, std::unique_ptr<PointCloudSensor>> pointcloud_sensors_;

  // Pose buffer
  std::unique_ptr<RingBuffer<geometry_msgs::msg::PoseStamped>> pose_buffer_;

  // RGB buffer (unified - both compressed and raw images converted to rgb8 and stored here)
  std::unique_ptr<RingBuffer<sensor_msgs::msg::Image>> rgb_raw_buffer_;

  // PointCloud buffer
  std::unique_ptr<RingBuffer<sensor_msgs::msg::PointCloud2>> pc_buffer_;

  // CameraInfo (cached once)
  sensor_msgs::msg::CameraInfo camera_info_;
  bool has_camera_info_{false};
  std::string camera_frame_id_;  // Cached from CameraInfo header

  // TF
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_ptr_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // Cached transforms (frame pairs)
  bool has_pose_T_base_{false};
  geometry_msgs::msg::TransformStamped pose_T_base_;
  std::string cached_pose_frame_pair_;

  bool has_base_T_camera_{false};
  geometry_msgs::msg::TransformStamped base_T_camera_;
  std::string cached_camera_frame_pair_;

  bool has_pointcloud_T_base_{false};
  geometry_msgs::msg::TransformStamped pointcloud_T_base_;
  std::string cached_pointcloud_frame_pair_;

  // Topic statistics trackers
  std::unique_ptr<TopicStatsTracker> pose_stats_;
  std::unique_ptr<TopicStatsTracker> rgb_stats_;
  std::unique_ptr<TopicStatsTracker> pc_stats_;

  // Python interpreter management
  // Interpreter lifecycle managed by SemMapWrapper (pybind11)
  // Initialization: py::initialize_interpreter(false) if not already initialized
  // Cleanup: Never call py::finalize_interpreter() - prevents double-free in component containers

  // Unified semantic map (Python embedding via pybind11)
  std::unique_ptr<SemMapWrapper> sem_map_;
  double semantic_rate_hz_{0.0};

  // Cached at init so the stats timer never acquires the GIL: a pipeline
  // thread stuck inside Python holding the GIL must not freeze the node's
  // callback group (all subscriptions share it).
  std::string cached_seg_layer_;
  std::string cached_backbone_;

  // Processing threads
  std::thread semantic_thread_;
  std::atomic<bool> semantic_thread_running_{false};
  uint64_t last_processed_rgb_stamp_ns_{0};
  std::atomic<uint64_t> pose_interp_fail_rgb_{0};

  std::thread pointcloud_thread_;
  std::atomic<bool> pointcloud_thread_running_{false};
  uint64_t last_processed_pc_stamp_ns_{0};
  std::atomic<uint64_t> pose_interp_fail_pc_{0};

  std::thread ros_publish_thread_;
  std::atomic<bool> ros_publish_thread_running_{false};
  uint64_t last_published_elevation_stamp_ns_{0};

  // Last published elevation map (for republishing when no new data available)
  std::map<std::string, std::vector<float>> last_published_elevation_layers_;
  int last_published_elevation_h_{0};
  int last_published_elevation_w_{0};
  uint64_t last_published_elevation_map_stamp_ns_{0};
  std::array<double, 3> last_published_elevation_map_center_{0.0, 0.0, 0.0};
  std::mutex last_elevation_map_mutex_;

  // Flag to track if we've received at least one valid elevation map result
  // Only publish GridMap/TF/pose after this is true
  bool has_received_valid_elevation_map_{false};

  // Map center tracking
  std::array<double, 3> map_center_{0.0, 0.0, 0.0};
  std::mutex map_center_mutex_;

  // Pre-allocated buffers (reused across iterations to avoid dynamic allocations)
  std::vector<float> traversability_buffer_;
  std::vector<uint16_t> traversability_u16_buffer_;
  std::vector<float> segmentation_buffer_;
  std::vector<uint8_t> segmentation_rgb_buffer_;
  std::vector<float> confidence_buffer_;
  std::vector<uint16_t> confidence_u16_buffer_;
  std::vector<float> pointcloud_buffer_;
  std::vector<float> pointcloud_transformed_buffer_;
  std::vector<float> elevation_buffer_;
  std::vector<float> variance_buffer_;
  std::vector<float> trail_buffer_;
  std::vector<uint8_t> image_data_buffer_;
  std::vector<std::array<uint8_t, 3>> segmentation_color_lut_;
  bool segmentation_lut_ready_{false};

  // Thread statistics (rolling window averages)
  std::map<std::string, ThreadStats> thread_stats_;

  // Stats publisher
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr stats_pub_;

  // Stats data (accessed from stats timer)
  /// @brief Status messages queued for the stats timer; @c mutex guards both fields.
  struct StatsData {
    std::mutex mutex;
    std::vector<std::pair<std::string, bool>> status_messages;  // (message, is_error)
  } stats_data_;
};

}  // namespace sem_map
