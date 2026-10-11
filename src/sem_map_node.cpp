// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
// Public header with declarations and utilities
#include "sem_map/sem_map_node.hpp"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace sem_map {

SemMapNode::SemMapNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("sem_map_node", rclcpp::NodeOptions(options)
                                       .allow_undeclared_parameters(true)
                                       .automatically_declare_parameters_from_overrides(true)) {
  // Get or declare parameters (compatible with auto-declared YAML)
  auto get_or_declare_str = [this](const char* name, const char* def) {
    if (this->has_parameter(name)) return this->get_parameter(name).as_string();
    return this->declare_parameter<std::string>(name, def);
  };
  auto get_or_declare_int = [this](const char* name, int def) {
    if (this->has_parameter(name)) return this->get_parameter(name).as_int();
    return this->declare_parameter<int>(name, def);
  };
  auto get_or_declare_double = [this](const char* name, double def) {
    if (this->has_parameter(name)) return this->get_parameter(name).as_double();
    return this->declare_parameter<double>(name, def);
  };
  auto get_or_declare_bool = [this](const char* name, bool def) {
    if (this->has_parameter(name)) return this->get_parameter(name).as_bool();
    return this->declare_parameter<bool>(name, def);
  };

  pose_topic_ = get_or_declare_str("pose_topic", "/zed/zed_node/pose");
  pointcloud_topic_ = get_or_declare_str("pointcloud_topic", "/rslidar_points");
  image_topic_ =
      get_or_declare_str("image_topic", "/zed/zed_node/left/image_rect_color/compressed");
  image_is_compressed_ = get_or_declare_bool("image_is_compressed", true);
  pose_is_odometry_ = get_or_declare_bool("pose_is_odometry", false);
  camera_info_topic_ = get_or_declare_str("camera_info_topic", "/zed/zed_node/left/camera_info");

  world_frame_ = get_or_declare_str("world_frame", "map");
  base_frame_ = get_or_declare_str("base_frame", "zed_camera_link");
  pose_frame_ = get_or_declare_str("pose_frame", "zed_camera_center");

  rgb_buffer_size_ = static_cast<size_t>(get_or_declare_int("rgb_buffer_size", 100));
  pointcloud_buffer_size_ = static_cast<size_t>(get_or_declare_int("pointcloud_buffer_size", 100));
  pose_buffer_size_ = static_cast<size_t>(get_or_declare_int("pose_buffer_size", 1000));

  pointcloud_processing_rate_hz_ = get_or_declare_double("pointcloud_processing_rate_hz", 10.0);
  rgb_processing_rate_hz_ = get_or_declare_double("rgb_processing_rate_hz", 10.0);

  map_resolution_ = get_or_declare_double("resolution", 0.10);
  map_size_ = get_or_declare_double("map_size", 20.0);
  map_backend_ = get_or_declare_str("backend", "cupy");
  map_precision_ = get_or_declare_str("precision", "float16");
  semantic_bev_layer_name_ = get_or_declare_str("semantic_bev_layer_name", "trail");

  bool enable_height_map = get_or_declare_bool("enable_height_map", true);
  bool enable_bev_fusion = get_or_declare_bool("enable_bev_fusion", true);
  publish_traversability_ = get_or_declare_bool("publish_traversability", true);
  publish_segmentation_ = get_or_declare_bool("publish_segmentation", true);
  publish_confidence_ = get_or_declare_bool("publish_confidence", true);
  publish_custom_elevation_ = get_or_declare_bool("publish_custom_elevation", false);
  custom_elevation_ = static_cast<float>(get_or_declare_double("custom_elevation", 0.0));

  // Store feature flags as member variables
  // enable_height_map is the top-level switch: when false, no pose/pointcloud/BEV fusion needed
  enable_height_map_ = enable_height_map;
  enable_bev_fusion_ = enable_height_map ? enable_bev_fusion : false;

  // Initialize buffers with parameter values
  rgb_raw_buffer_ = std::make_unique<RingBuffer<sensor_msgs::msg::Image>>(rgb_buffer_size_);

  if (enable_height_map) {
    // Pose, pointcloud buffers and TF only needed when height mapping is enabled
    pose_buffer_ = std::make_unique<RingBuffer<geometry_msgs::msg::PoseStamped>>(pose_buffer_size_);
    pc_buffer_ =
        std::make_unique<RingBuffer<sensor_msgs::msg::PointCloud2>>(pointcloud_buffer_size_);

    // TF buffer/listener for frame transforms
    tf_buffer_ptr_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_ptr_);

    // Pose subscription
    pose_stats_ = std::make_unique<TopicStatsTracker>(this, pose_topic_, "pose");
    if (pose_is_odometry_) {
      pose_odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
          pose_topic_, rclcpp::SensorDataQoS(),
          std::bind(&SemMapNode::on_pose_odom, this, std::placeholders::_1, std::placeholders::_2));
    } else {
      pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
          pose_topic_, rclcpp::SensorDataQoS(),
          std::bind(&SemMapNode::on_pose, this, std::placeholders::_1, std::placeholders::_2));
    }
  }

  // Conditionally subscribe to pointcloud if height mapping is enabled
  if (enable_height_map) {
    pc_stats_ = std::make_unique<TopicStatsTracker>(this, pointcloud_topic_, "pointcloud");
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, rclcpp::SensorDataQoS(),
        std::bind(&SemMapNode::on_pointcloud_single, this, std::placeholders::_1,
                  std::placeholders::_2));
  }

  rgb_stats_ = std::make_unique<TopicStatsTracker>(this, image_topic_, "rgb");
  if (image_is_compressed_) {
    // Subscribe to compressed and convert to raw immediately
    rgb_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&SemMapNode::on_rgb_compressed, this, std::placeholders::_1));
  } else {
    // Subscribe to raw directly
    rgb_raw_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic_, rclcpp::SensorDataQoS(),
        std::bind(&SemMapNode::on_rgb_raw, this, std::placeholders::_1, std::placeholders::_2));
  }

  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic_, rclcpp::SensorDataQoS(),
      std::bind(&SemMapNode::on_cam_info_single, this, std::placeholders::_1));

  // Publishers
  if (publish_traversability_) {
    traversability_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/sem_map/traversability",
                                                                          rclcpp::SensorDataQoS());
  }
  if (publish_segmentation_) {
    segmentation_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/sem_map/segmentation",
                                                                        rclcpp::SensorDataQoS());
  }
  if (publish_confidence_) {
    confidence_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/sem_map/confidence",
                                                                      rclcpp::SensorDataQoS());
    RCLCPP_INFO(this->get_logger(), "Publishing confidence map on /sem_map/confidence");
  }
  auto grid_map_topic = get_or_declare_str("grid_map_topic", "/sem_map/grid_map");
  grid_map_pub_ =
      this->create_publisher<grid_map_msgs::msg::GridMap>(grid_map_topic, rclcpp::SensorDataQoS());
  map_center_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "/sem_map/map_center", rclcpp::SensorDataQoS());

  // TF broadcaster
  tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

  // ROS publishing rate
  ros_publish_rate_hz_ = get_or_declare_double("ros_publish_rate_hz", 30.0);

  // Stats publisher
  stats_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/sem_map/stats", 10);

  // Service: reset the elevation map (clears all layers in the elevation worker)
  reset_elevation_map_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "~/reset_elevation_map",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
             std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
        const bool ok = sem_map_ && sem_map_->reset_elevation_map();
        response->success = ok;
        response->message = ok ? "reset signal sent to elevation worker"
                               : "failed to signal reset (multiprocessing not enabled?)";
        if (ok) {
          RCLCPP_INFO(this->get_logger(), "Elevation map reset requested via service");
        } else {
          RCLCPP_WARN(this->get_logger(), "Elevation map reset request failed: %s",
                      response->message.c_str());
        }
      });

  // Print startup configuration
  RCLCPP_INFO(this->get_logger(), "=== SemMapNode Configuration ===");
  RCLCPP_INFO(this->get_logger(), "Pose topic: %s", pose_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "PointCloud topic: %s", pointcloud_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "Image topic: %s (%s)", image_topic_.c_str(),
              image_is_compressed_ ? "compressed" : "raw");
  RCLCPP_INFO(this->get_logger(), "CameraInfo topic: %s", camera_info_topic_.c_str());
  RCLCPP_INFO(this->get_logger(), "World frame: %s", world_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "Base frame: %s", base_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "Pose frame: %s", pose_frame_.c_str());
  RCLCPP_INFO(this->get_logger(), "RGB buffer size: %zu", rgb_buffer_size_);
  RCLCPP_INFO(this->get_logger(), "PointCloud buffer size: %zu", pointcloud_buffer_size_);
  RCLCPP_INFO(this->get_logger(), "Pose buffer size: %zu", pose_buffer_size_);
  RCLCPP_INFO(this->get_logger(), "Message throttle time: %d ms", MESSAGE_THROTTLE_TIME_MS);
  RCLCPP_INFO(this->get_logger(), "Statistics window size: 100");
  RCLCPP_INFO(this->get_logger(), "Grid map topic: %s", grid_map_topic.c_str());
  RCLCPP_INFO(this->get_logger(),
              "Processing rates: pointcloud=%.1f Hz, rgb=%.1f Hz, publish=%.1f Hz",
              pointcloud_processing_rate_hz_, rgb_processing_rate_hz_, ros_publish_rate_hz_);
  RCLCPP_INFO(this->get_logger(), "Map: size=%.1f m, resolution=%.2f m, backend=%s, precision=%s",
              map_size_, map_resolution_, map_backend_.c_str(), map_precision_.c_str());
  RCLCPP_INFO(this->get_logger(), "Semantic BEV layer: %s", semantic_bev_layer_name_.c_str());
  RCLCPP_INFO(this->get_logger(), "Elevation mapping enabled: %s",
              enable_height_map ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "BEV fusion enabled: %s", enable_bev_fusion ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Publish traversability: %s",
              publish_traversability_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Publish segmentation: %s",
              publish_segmentation_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Publish confidence: %s", publish_confidence_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "=== SemMapNode Started Successfully ===");

  // Periodic statistics printer and publisher (FPS per topic)
  using namespace std::chrono_literals;
  stats_timer_ = this->create_wall_timer(1000ms, [this]() {
    // Build and publish DiagnosticArray. Severity levels are all OK for now;
    // thresholds will be added in a follow-up.
    diagnostic_msgs::msg::DiagnosticArray stats_msg;
    stats_msg.header.stamp = this->now();
    stats_msg.header.frame_id = "sem_map_stats";

    auto fmt = [](double v) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.3f", v);
      return std::string(buf);
    };
    auto add_topic_status = [&](const std::string& topic, double hz, double latency_ms) {
      diagnostic_msgs::msg::DiagnosticStatus s;
      s.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      s.name = "topic: " + topic;
      diagnostic_msgs::msg::KeyValue kv_hz;
      kv_hz.key = "hz";
      kv_hz.value = fmt(hz);
      s.values.push_back(kv_hz);
      diagnostic_msgs::msg::KeyValue kv_lat;
      kv_lat.key = "latency_ms";
      kv_lat.value = fmt(latency_ms);
      s.values.push_back(kv_lat);
      stats_msg.status.push_back(s);
    };

    std::lock_guard<std::mutex> lock(stats_data_.mutex);

    if (pose_stats_) {
      add_topic_status(pose_topic_, pose_stats_->get_fps(), pose_stats_->get_avg_latency());
    }
    add_topic_status(image_topic_, rgb_stats_->get_fps(), rgb_stats_->get_avg_latency());
    if (pc_stats_) {
      add_topic_status(pointcloud_topic_, pc_stats_->get_fps(), pc_stats_->get_avg_latency());
    }

    // Config snapshot (static-ish, but bundled in every tick so a single
    // `ros2 topic echo --once` is self-contained).
    {
      diagnostic_msgs::msg::DiagnosticStatus s;
      s.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      s.name = "config";
      auto add_kv = [&](const std::string& k, const std::string& v) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = k;
        kv.value = v;
        s.values.push_back(kv);
      };
      add_kv("pose_topic", pose_topic_);
      add_kv("pointcloud_topic", pointcloud_topic_);
      add_kv("image_topic", image_topic_);
      add_kv("image_is_compressed", image_is_compressed_ ? "true" : "false");
      add_kv("camera_info_topic", camera_info_topic_);
      add_kv("world_frame", world_frame_);
      add_kv("base_frame", base_frame_);
      add_kv("pose_frame", pose_frame_);
      add_kv("rgb_buffer_size", std::to_string(rgb_buffer_size_));
      add_kv("pointcloud_buffer_size", std::to_string(pointcloud_buffer_size_));
      add_kv("pose_buffer_size", std::to_string(pose_buffer_size_));
      add_kv("pointcloud_processing_rate_hz", fmt(pointcloud_processing_rate_hz_));
      add_kv("rgb_processing_rate_hz", fmt(rgb_processing_rate_hz_));
      add_kv("ros_publish_rate_hz", fmt(ros_publish_rate_hz_));
      add_kv("publish_traversability", publish_traversability_ ? "true" : "false");
      add_kv("publish_segmentation", publish_segmentation_ ? "true" : "false");
      add_kv("publish_confidence", publish_confidence_ ? "true" : "false");
      add_kv("map_resolution", fmt(map_resolution_));
      add_kv("map_size", fmt(map_size_));
      add_kv("map_backend", map_backend_);
      add_kv("map_precision", map_precision_);
      add_kv("semantic_bev_layer", semantic_bev_layer_name_);
      add_kv("enable_bev_fusion", enable_bev_fusion_ ? "true" : "false");
      add_kv("enable_height_map", enable_height_map_ ? "true" : "false");
      // Cached at init: must not call into Python (GIL) from this callback,
      // or a stuck pipeline thread would freeze the whole callback group.
      if (!cached_seg_layer_.empty()) {
        add_kv("seg_layer", cached_seg_layer_);
        add_kv("backbone", cached_backbone_);
      }
      stats_msg.status.push_back(s);
    }

    // Threads
    for (auto& [name, tracker] : thread_stats_) {
      diagnostic_msgs::msg::DiagnosticStatus s;
      s.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      s.name = "thread: " + name;
      diagnostic_msgs::msg::KeyValue kv_avg;
      kv_avg.key = "avg_ms";
      kv_avg.value = fmt(tracker.get_avg_ms());
      s.values.push_back(kv_avg);
      diagnostic_msgs::msg::KeyValue kv_n;
      kv_n.key = "samples";
      kv_n.value = std::to_string(tracker.get_count());
      s.values.push_back(kv_n);
      stats_msg.status.push_back(s);
    }

    // Map center (read under its own mutex)
    {
      diagnostic_msgs::msg::DiagnosticStatus s;
      s.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      s.name = "map";
      std::lock_guard<std::mutex> mc_lock(map_center_mutex_);
      for (size_t i = 0; i < 3; ++i) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = (i == 0 ? "center_x" : (i == 1 ? "center_y" : "center_z"));
        kv.value = fmt(map_center_[i]);
        s.values.push_back(kv);
      }
      stats_msg.status.push_back(s);
    }

    // Accumulated status messages (one DiagnosticStatus, one KV per message)
    {
      diagnostic_msgs::msg::DiagnosticStatus s;
      s.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
      s.name = "status";
      int idx = 0;
      for (const auto& [text, is_error] : stats_data_.status_messages) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = "msg_" + std::to_string(idx++) + (is_error ? "_err" : "");
        kv.value = text;
        s.values.push_back(kv);
      }
      stats_msg.status.push_back(s);
    }
    stats_data_.status_messages.clear();

    stats_pub_->publish(stats_msg);

    // Log topic statistics (wall-time throttled, every MESSAGE_THROTTLE_TIME_MS)
    // Using a counter since this wall_timer fires every 1s, avoiding sim_time clock issues
    static int stats_log_counter = 0;
    if (++stats_log_counter >= (MESSAGE_THROTTLE_TIME_MS / 1000)) {
      stats_log_counter = 0;

      if (rgb_stats_) {
        RCLCPP_INFO(this->get_logger(), "[stats] rgb fps=%.2f total=%lu window=%zu latency=%.3fms",
                    rgb_stats_->get_fps(),
                    static_cast<unsigned long>(rgb_stats_->get_total_count()),
                    rgb_stats_->get_window_size(), rgb_stats_->get_avg_latency());
      }

      if (pc_stats_) {
        RCLCPP_INFO(this->get_logger(),
                    "[stats] pointcloud fps=%.2f total=%lu window=%zu latency=%.3fms",
                    pc_stats_->get_fps(), static_cast<unsigned long>(pc_stats_->get_total_count()),
                    pc_stats_->get_window_size(), pc_stats_->get_avg_latency());
      }

      if (pose_stats_) {
        RCLCPP_INFO(this->get_logger(), "[stats] pose fps=%.2f total=%lu window=%zu latency=%.3fms",
                    pose_stats_->get_fps(),
                    static_cast<unsigned long>(pose_stats_->get_total_count()),
                    pose_stats_->get_window_size(), pose_stats_->get_avg_latency());
      }

      auto rgb_interp_fail = pose_interp_fail_rgb_.load(std::memory_order_relaxed);
      auto pc_interp_fail = pose_interp_fail_pc_.load(std::memory_order_relaxed);
      if (rgb_interp_fail > 0 || pc_interp_fail > 0) {
        RCLCPP_INFO(this->get_logger(), "[pose_interp] failures: rgb=%lu pointcloud=%lu total=%lu",
                    static_cast<unsigned long>(rgb_interp_fail),
                    static_cast<unsigned long>(pc_interp_fail),
                    static_cast<unsigned long>(rgb_interp_fail + pc_interp_fail));
      }
    }

    // Publish diagnostic messages
    if (rgb_stats_) rgb_stats_->publish_stats();
    if (pc_stats_) pc_stats_->publish_stats();
    if (pose_stats_) pose_stats_->publish_stats();
  });

  // Semantic pipeline processing rate from rgb_processing_rate_hz
  semantic_rate_hz_ = std::max(1e-3, rgb_processing_rate_hz_);

  // Pre-allocate buffers to avoid dynamic allocations in hot paths
  using namespace buffer_sizes;
  traversability_buffer_.reserve(MAX_TRAVERSABILITY_PIXELS);
  traversability_u16_buffer_.reserve(MAX_TRAVERSABILITY_PIXELS);
  segmentation_buffer_.reserve(MAX_TRAVERSABILITY_PIXELS);
  segmentation_rgb_buffer_.reserve(MAX_TRAVERSABILITY_PIXELS * 3);
  pointcloud_buffer_.reserve(MAX_POINTCLOUD_XYZ);
  pointcloud_transformed_buffer_.reserve(MAX_POINTCLOUD_XYZ);
  elevation_buffer_.reserve(MAX_ELEVATION_MAP_CELLS);
  variance_buffer_.reserve(MAX_ELEVATION_MAP_CELLS);
  trail_buffer_.reserve(MAX_ELEVATION_MAP_CELLS);
  image_data_buffer_.reserve(MAX_IMAGE_BYTES);
  RCLCPP_INFO(this->get_logger(), "Pre-allocated buffers (trav=%zu, pc=%zu, elev=%zu)",
              MAX_TRAVERSABILITY_PIXELS, MAX_POINTCLOUD_XYZ, MAX_ELEVATION_MAP_CELLS);

  // Initialize semantic pipeline
  // Python interpreter lifecycle: py::initialize_interpreter(false) if needed, never finalize
  // This pattern is safe for ROS2 components to avoid double-free on unload
  try {
    sem_map_ = std::make_unique<SemMapWrapper>();
    sem_map_->initialize(this->get_node_parameters_interface());

    // Cache diagnostic strings once so the stats timer never needs the GIL
    cached_seg_layer_ = sem_map_->get_param_string("seg_layer");
    cached_backbone_ = sem_map_->get_param_string("backbone_name");

    if (publish_segmentation_) {
      segmentation_lut_ready_ = sem_map_->get_segmentation_color_lut(segmentation_color_lut_);
      if (!segmentation_lut_ready_) {
        RCLCPP_WARN(this->get_logger(),
                    "Segmentation color LUT unavailable, disabling segmentation publisher");
        publish_segmentation_ = false;
        segmentation_pub_.reset();
      }
    }

    // Start semantic processing thread
    semantic_thread_running_ = true;
    semantic_thread_ = std::thread(&SemMapNode::semantic_processing_loop, this);
    RCLCPP_INFO(this->get_logger(), "Semantic pipeline initialized (pybind11), rate=%.2f Hz",
                semantic_rate_hz_);

    // Conditionally start pointcloud thread if height mapping enabled
    if (enable_height_map) {
      pointcloud_thread_running_ = true;
      pointcloud_thread_ = std::thread(&SemMapNode::pointcloud_processing_loop, this);
      RCLCPP_INFO(this->get_logger(), "Pointcloud processing thread started, rate=%.2f Hz",
                  pointcloud_processing_rate_hz_);
    } else {
      RCLCPP_INFO(this->get_logger(), "Height mapping disabled - pointcloud thread not started");
    }

    // Start ROS publishing thread:
    // - Multiprocessing: always (publishes semantic + elevation results)
    // - Legacy: only if height map enabled (semantic publishing stays in RGB thread)
    if (sem_map_->is_multiprocessing_enabled() || enable_height_map) {
      ros_publish_thread_running_ = true;
      ros_publish_thread_ = std::thread(&SemMapNode::ros_publishing_loop, this);
      RCLCPP_INFO(this->get_logger(), "ROS publishing thread started, rate=%.2f Hz",
                  ros_publish_rate_hz_);
    }

  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize semantic pipeline: %s", e.what());
  }

  // Register pre-shutdown callback to stop threads before destructor
  // This ensures graceful cleanup within ROS2's shutdown timeout
  // Pattern based on ZED camera component
  using rclcpp::contexts::get_global_default_context;
  get_global_default_context()->add_pre_shutdown_callback([this]() {
    RCLCPP_DEBUG(this->get_logger(), "🛑 Pre-shutdown: stopping threads...");
    stopThreads();
  });
}

void SemMapNode::stopThreads() {
  // Signal all threads to stop
  semantic_thread_running_ = false;
  pointcloud_thread_running_ = false;
  ros_publish_thread_running_ = false;

  // Close Python pipeline first (signals worker processes to stop)
  // This is critical: Python workers must be signaled before we join C++ threads
  try {
    if (sem_map_) {
      sem_map_->close();
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN(this->get_logger(), "Error closing Python pipeline: %s", e.what());
  }

  // Now join C++ threads (they should exit quickly now)
  if (semantic_thread_.joinable()) {
    semantic_thread_.join();
  }

  if (pointcloud_thread_.joinable()) {
    pointcloud_thread_.join();
  }

  if (ros_publish_thread_.joinable()) {
    ros_publish_thread_.join();
  }
}

SemMapNode::~SemMapNode() {
  // Note: In ROS2 component containers with SIGKILL shutdown, this destructor typically never runs.
  // All cleanup is handled by the pre-shutdown callback registered in constructor.
  // This destructor is kept as a failsafe for standalone/test scenarios.
  RCLCPP_DEBUG(this->get_logger(), "SemMapNode destructor called");
}

void SemMapNode::semantic_processing_loop() {
  const auto period = std::chrono::duration<double>(1.0 / std::max(1e-3, semantic_rate_hz_));

  // Hold a pending message locally to avoid race between front()/pop_front() and push().
  // On retry (TOO_NEW, INSUFFICIENT_POSES), the message stays here instead of the buffer.
  sensor_msgs::msg::Image pending_msg;
  bool has_pending = false;
  uint64_t pending_jump_gen = 0;
  auto pending_since = std::chrono::steady_clock::now();

  while (semantic_thread_running_ && rclcpp::ok()) {
    auto loop_start = std::chrono::steady_clock::now();

    // Drop a pre-time-jump pending message so we don't sit on it until bag
    // time naturally catches up after a rosbag restart. A pre-restart frame's
    // stamp is in the future relative to the restarted clock, so it would be
    // permanently TOO_NEW for pose interpolation otherwise.
    if (has_pending && pending_jump_gen != time_jump_counter_.load(std::memory_order_relaxed)) {
      RCLCPP_WARN(this->get_logger(),
                  "[semantic] Dropping pending RGB frame from before time jump");
      has_pending = false;
    }

    // Try to get a message: either from pending or by popping the buffer
    if (!has_pending) {
      if (rgb_raw_buffer_ && has_camera_info_ && rgb_raw_buffer_->try_pop_front(pending_msg)) {
        has_pending = true;
        pending_jump_gen = time_jump_counter_.load(std::memory_order_relaxed);
        pending_since = std::chrono::steady_clock::now();
      }
    }

    if (has_pending) {
      const auto& msg = pending_msg;
      const uint64_t stamp_ns =
          static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ULL + msg.header.stamp.nanosec;
      if (last_processed_rgb_stamp_ns_ != stamp_ns) {
        // Check pose interpolation status first (only if BEV fusion is enabled)
        geometry_msgs::msg::PoseStamped robot_pose;
        PoseInterpolationStatus interp_status = PoseInterpolationStatus::SUCCESS;

        if (has_base_T_camera_ && enable_bev_fusion_) {
          interp_status = interpolate_pose(stamp_ns, robot_pose);

          if (interp_status == PoseInterpolationStatus::TOO_NEW) {
            // RGB is in the future - keep pending, wait for pose to arrive.
            // Escape hatch: if poses never catch up (stalled pose source, clock
            // offset between camera and pose clocks), drop the frame after a
            // bounded wait instead of blocking the pipeline forever.
            if (std::chrono::steady_clock::now() - pending_since >
                std::chrono::duration<double>(MAX_PENDING_HOLD_SEC)) {
              RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                   "[semantic] RGB timestamp (%.3fs) still newer than newest pose "
                                   "after %.1fs, dropping "
                                   "(pose source stalled or clock mismatch?)",
                                   stamp_ns / 1e9, MAX_PENDING_HOLD_SEC);
              pose_interp_fail_rgb_.fetch_add(1, std::memory_order_relaxed);
              has_pending = false;
              continue;
            }
            RCLCPP_DEBUG_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "[semantic] RGB timestamp (%.3fs) is in the future, waiting for pose...",
                stamp_ns / 1e9);
            last_processed_rgb_stamp_ns_ = 0;
            auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
            if (loop_elapsed < period) {
              std::this_thread::sleep_for(period - loop_elapsed);
            }
            continue;
          } else if (interp_status == PoseInterpolationStatus::TOO_OLD) {
            // RGB is too old - drop it (we'll never get a pose for it)
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "[semantic] RGB timestamp (%.3fs) is too old, dropping",
                                 stamp_ns / 1e9);
            pose_interp_fail_rgb_.fetch_add(1, std::memory_order_relaxed);
            has_pending = false;
            continue;
          } else if (interp_status == PoseInterpolationStatus::INSUFFICIENT_POSES) {
            // Not enough poses yet - keep pending, wait (bounded, same escape as TOO_NEW)
            if (std::chrono::steady_clock::now() - pending_since >
                std::chrono::duration<double>(MAX_PENDING_HOLD_SEC)) {
              RCLCPP_WARN_THROTTLE(
                  this->get_logger(), *this->get_clock(), 5000,
                  "[semantic] Insufficient poses for RGB timestamp (%.3fs) after %.1fs, dropping",
                  stamp_ns / 1e9, MAX_PENDING_HOLD_SEC);
              pose_interp_fail_rgb_.fetch_add(1, std::memory_order_relaxed);
              has_pending = false;
              continue;
            }
            RCLCPP_DEBUG_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "[semantic] Insufficient poses for RGB timestamp (%.3fs), waiting...",
                stamp_ns / 1e9);
            last_processed_rgb_stamp_ns_ = 0;
            auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
            if (loop_elapsed < period) {
              std::this_thread::sleep_for(period - loop_elapsed);
            }
            continue;
          }
          // If SUCCESS, continue to processing below with robot_pose already populated
        }

        last_processed_rgb_stamp_ns_ = stamp_ns;
        auto proc_start = std::chrono::steady_clock::now();
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                              "[semantic] Processing RGB frame, buffer_size=%zu",
                              rgb_raw_buffer_->size());

        // Image is already in rgb8 format from callbacks
        const int height = static_cast<int>(msg.height);
        const int width = static_cast<int>(msg.width);
        const int channels = 3;

        try {
          // Compute camera pose for traversability fusion
          Eigen::Matrix3d* pose_R_ptr = nullptr;
          Eigen::Vector3d* pose_t_ptr = nullptr;
          Eigen::Matrix3d pose_R;
          Eigen::Vector3d pose_t;

          // Only compute pose if BEV fusion is enabled
          if (has_base_T_camera_ && enable_bev_fusion_) {
            // robot_pose already populated from status check above
            if (interp_status == PoseInterpolationStatus::SUCCESS) {
              // Extract robot pose (base in world frame)
              Eigen::Vector3d base_position(robot_pose.pose.position.x, robot_pose.pose.position.y,
                                            robot_pose.pose.position.z);
              Eigen::Quaterniond base_quat(
                  robot_pose.pose.orientation.w, robot_pose.pose.orientation.x,
                  robot_pose.pose.orientation.y, robot_pose.pose.orientation.z);
              Eigen::Matrix3d base_R = base_quat.toRotationMatrix();

              // Extract base-to-camera transform
              const auto& t_b2c = base_T_camera_.transform.translation;
              const auto& q_b2c = base_T_camera_.transform.rotation;
              Eigen::Quaterniond camera_quat_in_base(q_b2c.w, q_b2c.x, q_b2c.y, q_b2c.z);
              Eigen::Matrix3d camera_R_in_base = camera_quat_in_base.toRotationMatrix();
              Eigen::Vector3d camera_t_in_base(t_b2c.x, t_b2c.y, t_b2c.z);

              // Compute camera pose in world frame
              // camera_position_world = base_position + base_R @ camera_t_in_base
              // camera_R_world = base_R @ camera_R_in_base
              pose_t = base_position + base_R * camera_t_in_base;
              pose_R = base_R * camera_R_in_base;

              pose_R_ptr = &pose_R;
              pose_t_ptr = &pose_t;

              RCLCPP_DEBUG_THROTTLE(
                  this->get_logger(), *this->get_clock(), 5000,
                  "[semantic] Passing camera pose to process_frame: pos=[%.2f, %.2f, %.2f]",
                  pose_t.x(), pose_t.y(), pose_t.z());
            } else {
              // This should not happen since we checked status above, but handle gracefully
              pose_interp_fail_rgb_.fetch_add(1, std::memory_order_relaxed);
              RCLCPP_WARN_THROTTLE(
                  this->get_logger(), *this->get_clock(), 5000,
                  "[semantic] Failed to interpolate pose for RGB timestamp (status=%d), skipping "
                  "BEV fusion (traversability generation will continue)",
                  static_cast<int>(interp_status));
            }
          } else if (!enable_bev_fusion_) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                  "[semantic] BEV fusion disabled, skipping pose computation");
          } else {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                  "[semantic] No base_T_camera transform, no fusion");
          }

          // Note: If pose interpolation failed, pose_R_ptr and pose_t_ptr remain nullptr,
          // so BEV fusion will be skipped but traversability generation will continue

          // Check if multiprocessing mode is enabled
          bool multiprocessing = sem_map_->is_multiprocessing_enabled();

          if (multiprocessing) {
            // ========== Multiprocessing Mode ==========
            // Write RGB to buffer for semantic worker
            // Extract camera intrinsics from camera_info_
            Eigen::Matrix3d K_eigen;
            if (has_camera_info_) {
              K_eigen << camera_info_.k[0], camera_info_.k[1], camera_info_.k[2], camera_info_.k[3],
                  camera_info_.k[4], camera_info_.k[5], camera_info_.k[6], camera_info_.k[7],
                  camera_info_.k[8];
            } else {
              // Fallback to identity if camera info not yet received
              K_eigen = Eigen::Matrix3d::Identity();
            }

            bool write_ok = sem_map_->write_rgb_to_buffer(
                msg.data.data(), height, width, channels, msg.header.frame_id, stamp_ns,
                pose_R_ptr ? *pose_R_ptr : Eigen::Matrix3d::Identity(),
                pose_t_ptr ? *pose_t_ptr : Eigen::Vector3d::Zero(), K_eigen);

            if (!write_ok) {
              RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                   "[semantic] Failed to write RGB to buffer");
            }
          } else {
            // ========== Legacy Mode ==========
            // Process frame (Python TensorRT inference + BEV fusion if enabled)
            sem_map_->process_frame(const_cast<uint8_t*>(msg.data.data()), height, width, channels,
                                    std::string(msg.header.frame_id.c_str()), stamp_ns, pose_R_ptr,
                                    pose_t_ptr);
          }

          // Legacy mode: get semantic results and publish here
          // Multiprocessing mode: semantic publishing moved to ros_publishing_loop
          if (!multiprocessing) {
            traversability_buffer_.clear();
            segmentation_buffer_.clear();
            int trav_h = 0, trav_w = 0;
            bool has_semantic = sem_map_->get_last_semantic_results(
                traversability_buffer_, segmentation_buffer_, trav_h, trav_w);

            if (has_semantic) {
              size_t expected_size = static_cast<size_t>(trav_h * trav_w);

              if (publish_traversability_ && traversability_pub_) {
                if (traversability_u16_buffer_.size() != expected_size) {
                  traversability_u16_buffer_.resize(expected_size);
                }

                const float* src = traversability_buffer_.data();
                uint16_t* dst = traversability_u16_buffer_.data();
                for (size_t i = 0; i < expected_size; ++i) {
                  float v = std::clamp(src[i], 0.0f, 1.0f);
                  dst[i] = static_cast<uint16_t>(v * 65535.0f + 0.5f);
                }

                sensor_msgs::msg::Image out;
                out.header = msg.header;
                out.height = static_cast<uint32_t>(trav_h);
                out.width = static_cast<uint32_t>(trav_w);
                out.encoding = "mono16";
                out.is_bigendian = false;
                out.step = out.width * 2;
                size_t required_bytes = traversability_u16_buffer_.size() * sizeof(uint16_t);
                if (out.data.size() != required_bytes) {
                  out.data.resize(required_bytes);
                }
                std::memcpy(out.data.data(), traversability_u16_buffer_.data(), required_bytes);
                traversability_pub_->publish(out);
              }

              if (publish_segmentation_ && segmentation_pub_ && segmentation_lut_ready_) {
                if (segmentation_buffer_.size() >= expected_size) {
                  segmentation_rgb_buffer_.resize(expected_size * 3);
                  const size_t lut_size = segmentation_color_lut_.size();
                  const std::array<uint8_t, 3> transparent_color{0, 0, 0};

                  for (size_t i = 0; i < expected_size; ++i) {
                    float value = segmentation_buffer_[i];
                    int label = static_cast<int>(std::round(value));
                    bool out_of_range = (label < 0) || (static_cast<size_t>(label) >= lut_size);
                    if (out_of_range) {
                      label = 255;
                    }

                    const auto& color = out_of_range
                                            ? transparent_color
                                            : segmentation_color_lut_[static_cast<size_t>(label)];
                    size_t base = i * 3;
                    segmentation_rgb_buffer_[base + 0] = color[0];
                    segmentation_rgb_buffer_[base + 1] = color[1];
                    segmentation_rgb_buffer_[base + 2] = color[2];
                  }

                  sensor_msgs::msg::Image seg_msg;
                  seg_msg.header = msg.header;
                  seg_msg.height = static_cast<uint32_t>(trav_h);
                  seg_msg.width = static_cast<uint32_t>(trav_w);
                  seg_msg.encoding = "rgb8";
                  seg_msg.is_bigendian = false;
                  seg_msg.step = seg_msg.width * 3;
                  seg_msg.data = segmentation_rgb_buffer_;
                  segmentation_pub_->publish(seg_msg);
                }
              }
            }
          }

          // Record processing time
          auto proc_elapsed = std::chrono::steady_clock::now() - proc_start;
          thread_stats_["RGB Processing"].record(
              std::chrono::duration<double, std::milli>(proc_elapsed).count());

          has_pending = false;
        } catch (const std::exception& e) {
          std::string error_msg = std::string("Semantic processing error: ") + e.what();
          {
            std::lock_guard<std::mutex> lock(stats_data_.mutex);
            stats_data_.status_messages.push_back({error_msg, true});
          }
          RCLCPP_ERROR(this->get_logger(), "%s", error_msg.c_str());

          has_pending = false;
        }
      } else {
        // Timestamp already processed, discard
        has_pending = false;
      }
    } else {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                            "[semantic] Waiting: raw_buffer_size=%zu, has_cam_info=%d",
                            rgb_raw_buffer_ ? rgb_raw_buffer_->size() : 0, has_camera_info_);
    }

    // Sleep for remaining time to maintain rate
    auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
    if (loop_elapsed < period) {
      std::this_thread::sleep_for(period - loop_elapsed);
    }
  }
}

bool SemMapNode::find_closest_poses(uint64_t target_stamp_ns,
                                    geometry_msgs::msg::PoseStamped& before,
                                    geometry_msgs::msg::PoseStamped& after) {
  // Key extractor: pose -> timestamp in nanoseconds
  auto stamp_key = [](const geometry_msgs::msg::PoseStamped& pose) -> uint64_t {
    return static_cast<uint64_t>(pose.header.stamp.sec) * 1000000000ULL + pose.header.stamp.nanosec;
  };

  uint64_t oldest_stamp = 0, newest_stamp = 0;
  size_t buffer_size = 0;

  // O(log n) binary search under a single lock acquisition
  bool found = pose_buffer_->find_bracket(stamp_key, target_stamp_ns, before, after, oldest_stamp,
                                          newest_stamp, buffer_size);

  if (!found) {
    if (buffer_size < 2) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "[pose_interp] Insufficient poses in buffer: %zu (need >= 2)",
                           buffer_size);
    } else {
      double target_sec = target_stamp_ns / 1e9;
      double oldest_sec = oldest_stamp / 1e9;
      double newest_sec = newest_stamp / 1e9;
      double age_diff = target_sec - newest_sec;

      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "[pose_interp] Cannot bracket timestamp: target=%.3fs, buffer=[%.3fs, "
                           "%.3fs] (age_diff=%.3fs, buffer_size=%zu)",
                           target_sec, oldest_sec, newest_sec, age_diff, buffer_size);
    }
  }

  return found;
}

PoseInterpolationStatus SemMapNode::interpolate_pose(uint64_t target_stamp_ns,
                                                     geometry_msgs::msg::PoseStamped& result) {
  // Check if buffer is empty
  if (pose_buffer_->empty()) {
    return PoseInterpolationStatus::INSUFFICIENT_POSES;
  }

  // Check time bounds first (O(1) since buffer is chronologically ordered)
  // Safe copies avoid dangling references from front()/back()
  geometry_msgs::msg::PoseStamped oldest_pose, newest_pose;
  // back_copy implicitly checks non-empty; front uses same pattern
  if (!pose_buffer_->back_copy(newest_pose)) {
    return PoseInterpolationStatus::INSUFFICIENT_POSES;
  }
  // front is safe here: buffer is non-empty (back_copy succeeded) and only grows from push
  // (pop_front is never called on pose_buffer_). Using front() is acceptable for pose buffer
  // since the only writer is on_pose which only appends. But for consistency, copy it.
  oldest_pose = pose_buffer_->front();

  uint64_t oldest_stamp_ns = static_cast<uint64_t>(oldest_pose.header.stamp.sec) * 1000000000ULL +
                             oldest_pose.header.stamp.nanosec;
  uint64_t newest_stamp_ns = static_cast<uint64_t>(newest_pose.header.stamp.sec) * 1000000000ULL +
                             newest_pose.header.stamp.nanosec;

  // Check if target is too old
  if (target_stamp_ns < oldest_stamp_ns) {
    return PoseInterpolationStatus::TOO_OLD;
  }

  // Check if target is too new
  if (target_stamp_ns > newest_stamp_ns) {
    return PoseInterpolationStatus::TOO_NEW;
  }

  // Now try to find closest poses for interpolation
  geometry_msgs::msg::PoseStamped pose_before, pose_after;

  if (!find_closest_poses(target_stamp_ns, pose_before, pose_after)) {
    return PoseInterpolationStatus::INSUFFICIENT_POSES;
  }

  uint64_t t_before = static_cast<uint64_t>(pose_before.header.stamp.sec) * 1000000000ULL +
                      pose_before.header.stamp.nanosec;
  uint64_t t_after = static_cast<uint64_t>(pose_after.header.stamp.sec) * 1000000000ULL +
                     pose_after.header.stamp.nanosec;

  // If timestamps are the same, return one of them
  if (t_before == t_after) {
    result = pose_before;
    return PoseInterpolationStatus::SUCCESS;
  }

  // Compute interpolation factor
  double alpha =
      static_cast<double>(target_stamp_ns - t_before) / static_cast<double>(t_after - t_before);
  alpha = std::max(0.0, std::min(1.0, alpha));  // Clamp to [0,1]

  // Convert poses to Sophus SE3
  Eigen::Vector3d t_before_vec(pose_before.pose.position.x, pose_before.pose.position.y,
                               pose_before.pose.position.z);
  Eigen::Quaterniond q_before(pose_before.pose.orientation.w, pose_before.pose.orientation.x,
                              pose_before.pose.orientation.y, pose_before.pose.orientation.z);

  Eigen::Vector3d t_after_vec(pose_after.pose.position.x, pose_after.pose.position.y,
                              pose_after.pose.position.z);
  Eigen::Quaterniond q_after(pose_after.pose.orientation.w, pose_after.pose.orientation.x,
                             pose_after.pose.orientation.y, pose_after.pose.orientation.z);

  // Create SE3 transformations
  Sophus::SE3d T_before(q_before, t_before_vec);
  Sophus::SE3d T_after(q_after, t_after_vec);

  // Interpolate using Sophus (geodesic interpolation in SE3)
  // T_interp = T_before * exp(alpha * log(T_before^-1 * T_after))
  Sophus::SE3d T_delta = T_before.inverse() * T_after;
  Sophus::SE3d::Tangent delta_tangent = T_delta.log();
  Sophus::SE3d T_interp = T_before * Sophus::SE3d::exp(alpha * delta_tangent);

  // Extract interpolated pose
  result = pose_before;  // Copy header
  result.header.stamp.sec = target_stamp_ns / 1000000000ULL;
  result.header.stamp.nanosec = target_stamp_ns % 1000000000ULL;

  Eigen::Vector3d t_interp = T_interp.translation();
  Eigen::Quaterniond q_interp = T_interp.unit_quaternion();

  result.pose.position.x = t_interp.x();
  result.pose.position.y = t_interp.y();
  result.pose.position.z = t_interp.z();
  result.pose.orientation.w = q_interp.w();
  result.pose.orientation.x = q_interp.x();
  result.pose.orientation.y = q_interp.y();
  result.pose.orientation.z = q_interp.z();

  return PoseInterpolationStatus::SUCCESS;
}

bool SemMapNode::parse_pointcloud_xyz(const sensor_msgs::msg::PointCloud2& cloud,
                                      std::vector<float>& points) {
  points.clear();

  // Find x, y, z fields
  int x_offset = -1, y_offset = -1, z_offset = -1;
  sensor_msgs::msg::PointField::_datatype_type x_type = 0, y_type = 0, z_type = 0;

  for (const auto& field : cloud.fields) {
    if (field.name == "x") {
      x_offset = field.offset;
      x_type = field.datatype;
    } else if (field.name == "y") {
      y_offset = field.offset;
      y_type = field.datatype;
    } else if (field.name == "z") {
      z_offset = field.offset;
      z_type = field.datatype;
    }
  }

  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                          "PointCloud2 missing x, y, or z fields");
    return false;
  }

  // Reserve space
  size_t n_points = cloud.width * cloud.height;
  points.reserve(n_points * 3);

  // Extract XYZ data
  for (size_t i = 0; i < n_points; ++i) {
    size_t point_offset = i * cloud.point_step;

    float x, y, z;

    // Parse X
    if (x_type == sensor_msgs::msg::PointField::FLOAT32) {
      std::memcpy(&x, &cloud.data[point_offset + x_offset], sizeof(float));
    } else if (x_type == sensor_msgs::msg::PointField::FLOAT64) {
      double x_d;
      std::memcpy(&x_d, &cloud.data[point_offset + x_offset], sizeof(double));
      x = static_cast<float>(x_d);
    } else {
      continue;  // Skip unsupported type
    }

    // Parse Y
    if (y_type == sensor_msgs::msg::PointField::FLOAT32) {
      std::memcpy(&y, &cloud.data[point_offset + y_offset], sizeof(float));
    } else if (y_type == sensor_msgs::msg::PointField::FLOAT64) {
      double y_d;
      std::memcpy(&y_d, &cloud.data[point_offset + y_offset], sizeof(double));
      y = static_cast<float>(y_d);
    } else {
      continue;
    }

    // Parse Z
    if (z_type == sensor_msgs::msg::PointField::FLOAT32) {
      std::memcpy(&z, &cloud.data[point_offset + z_offset], sizeof(float));
    } else if (z_type == sensor_msgs::msg::PointField::FLOAT64) {
      double z_d;
      std::memcpy(&z_d, &cloud.data[point_offset + z_offset], sizeof(double));
      z = static_cast<float>(z_d);
    } else {
      continue;
    }

    // Skip invalid points (NaN or Inf)
    if (std::isnan(x) || std::isnan(y) || std::isnan(z) || std::isinf(x) || std::isinf(y) ||
        std::isinf(z)) {
      continue;
    }

    points.push_back(x);
    points.push_back(y);
    points.push_back(z);
  }

  return true;
}

std_msgs::msg::Float32MultiArray SemMapNode::create_float32_multiarray(
    const std::vector<float>& data, int rows, int cols) {
  std_msgs::msg::Float32MultiArray msg;

  // Set up dimensions
  msg.layout.dim.resize(2);

  // Row dimension (outer dimension) - GridMap uses column as outer
  msg.layout.dim[0].label = "column_index";
  msg.layout.dim[0].size = cols;
  msg.layout.dim[0].stride = rows * cols;

  // Column dimension (inner dimension) - GridMap uses row as inner
  msg.layout.dim[1].label = "row_index";
  msg.layout.dim[1].size = rows;
  msg.layout.dim[1].stride = rows;

  msg.layout.data_offset = 0;
  msg.data = data;

  return msg;
}

template <typename MessageType>
void SemMapNode::process_message(const MessageType& msg, const rclcpp::MessageInfo& /*info*/,
                                 TopicStatsTracker& stats_tracker,
                                 const std::string& /*msg_type*/) {
  rclcpp::Time now_time = this->get_clock()->now();
  uint64_t header_time_ns =
      static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ULL + msg.header.stamp.nanosec;
  double reception_header_ms =
      (static_cast<double>(now_time.nanoseconds()) - static_cast<double>(header_time_ns)) / 1e6;

  stats_tracker.record_message(reception_header_ms);
}

void SemMapNode::handle_pose(const geometry_msgs::msg::PoseStamped& msg,
                             const rclcpp::MessageInfo& info) {
  process_message(msg, info, *pose_stats_, "Pose");

  geometry_msgs::msg::PoseStamped pose_in_base = msg;

  if (pose_frame_ != base_frame_) {
    if (!has_pose_T_base_) {
      if (tf_buffer_ptr_->canTransform(pose_frame_, base_frame_, tf2::TimePointZero)) {
        pose_T_base_ =
            tf_buffer_ptr_->lookupTransform(pose_frame_, base_frame_, tf2::TimePointZero);
        has_pose_T_base_ = true;
        cached_pose_frame_pair_ = pose_frame_ + "->" + base_frame_;
        RCLCPP_INFO(this->get_logger(), "Cached pose_T_base for %s",
                    cached_pose_frame_pair_.c_str());
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), MESSAGE_THROTTLE_TIME_MS,
                             "Waiting for transform %s -> %s", pose_frame_.c_str(),
                             base_frame_.c_str());
      }
    }

    if (has_pose_T_base_) {
      try {
        tf2::Transform world_T_pose;
        tf2::fromMsg(msg.pose, world_T_pose);

        tf2::Transform pose_T_base_tf;
        tf2::fromMsg(pose_T_base_.transform, pose_T_base_tf);

        tf2::Transform world_T_base = world_T_pose * pose_T_base_tf;

        geometry_msgs::msg::Pose base_pose_msg;
        tf2::Vector3 origin = world_T_base.getOrigin();
        tf2::Quaternion rotation = world_T_base.getRotation();
        base_pose_msg.position.x = origin.x();
        base_pose_msg.position.y = origin.y();
        base_pose_msg.position.z = origin.z();
        base_pose_msg.orientation = tf2::toMsg(rotation);

        pose_in_base.pose = base_pose_msg;
        pose_in_base.header = msg.header;
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), MESSAGE_THROTTLE_TIME_MS,
                             "Failed to transform pose %s -> %s: %s", pose_frame_.c_str(),
                             base_frame_.c_str(), ex.what());
        return;
      }
    }
  }

  geometry_msgs::msg::PoseStamped back_pose;
  if (pose_buffer_->back_copy(back_pose)) {
    auto back_stamp = rclcpp::Time(back_pose.header.stamp);
    auto msg_stamp = rclcpp::Time(pose_in_base.header.stamp);

    // Detect backward time jump (rosbag loop / clock reset)
    if (msg_stamp < back_stamp - rclcpp::Duration::from_seconds(TIME_JUMP_THRESHOLD_SEC)) {
      RCLCPP_WARN(this->get_logger(),
                  "Time jump detected (%.3fs -> %.3fs, delta=%.3fs), flushing all buffers",
                  back_stamp.seconds(), msg_stamp.seconds(), (msg_stamp - back_stamp).seconds());

      pose_buffer_->clear();
      rgb_raw_buffer_->clear();
      if (pc_buffer_) pc_buffer_->clear();
      last_processed_rgb_stamp_ns_ = 0;
      last_processed_pc_stamp_ns_ = 0;
      // Tell worker loops to drop any in-hand pending_msg popped before the jump.
      time_jump_counter_.fetch_add(1, std::memory_order_relaxed);
    }

    // Monotonicity check — only push if newer than current back (re-check after potential clear)
    geometry_msgs::msg::PoseStamped current_back;
    if (!pose_buffer_->back_copy(current_back) ||
        msg_stamp > rclcpp::Time(current_back.header.stamp)) {
      pose_buffer_->push(pose_in_base);
    }
  } else {
    pose_buffer_->push(pose_in_base);
  }
}

void SemMapNode::on_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg,
                         const rclcpp::MessageInfo& info) {
  handle_pose(*msg, info);
}

void SemMapNode::on_pose_odom(const nav_msgs::msg::Odometry::SharedPtr msg,
                              const rclcpp::MessageInfo& info) {
  // header.frame_id is the odometry's parent (world) frame; the twist and
  // covariance are not used.
  geometry_msgs::msg::PoseStamped pose;
  pose.header = msg->header;
  pose.pose = msg->pose.pose;
  handle_pose(pose, info);
}

// Single-stream callbacks
void SemMapNode::on_cam_info_single(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  if (!has_camera_info_) {
    camera_info_ = *msg;
    has_camera_info_ = true;
    camera_frame_id_ = msg->header.frame_id;
    RCLCPP_INFO(this->get_logger(), "Received CameraInfo (single-stream)");
    RCLCPP_INFO(this->get_logger(), "   Image size: %dx%d", msg->width, msg->height);

    // Pass camera info to unified SemMap for traversability fusion
    // Note: In multiprocessing mode, camera intrinsics are passed per-frame in buffers,
    // so we skip this call (sem_map_ is empty in multiprocessing mode)
    if (sem_map_ && !sem_map_->is_multiprocessing_enabled()) {
      // Convert K array (std::array<double, 9>) to Eigen::Matrix3d
      Eigen::Matrix3d K;
      K << msg->k[0], msg->k[1], msg->k[2], msg->k[3], msg->k[4], msg->k[5], msg->k[6], msg->k[7],
          msg->k[8];

      // Convert D array to Eigen::VectorXd
      Eigen::VectorXd D = Eigen::Map<const Eigen::VectorXd>(msg->d.data(), msg->d.size());

      sem_map_->set_camera_info(K, D, msg->distortion_model, msg->width, msg->height);
      RCLCPP_INFO(this->get_logger(), "✅ Camera info passed to SemMap for traversability fusion");
    } else if (sem_map_ && sem_map_->is_multiprocessing_enabled()) {
      RCLCPP_INFO(this->get_logger(),
                  "ℹ️  Multiprocessing mode: camera intrinsics passed per-frame");
    }

    // Unsubscribe after first reception
    cam_info_sub_.reset();
  }
}

void SemMapNode::on_rgb_compressed(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
  if (rgb_stats_) {
    rclcpp::Time now_time = this->get_clock()->now();
    uint64_t header_time_ns =
        static_cast<uint64_t>(msg->header.stamp.sec) * 1000000000ULL + msg->header.stamp.nanosec;
    double reception_header_ms =
        (static_cast<double>(now_time.nanoseconds()) - static_cast<double>(header_time_ns)) / 1e6;
    rgb_stats_->record_message(reception_header_ms);
  }

  // Cache base->camera once (only needed for BEV fusion / height mapping)
  if (enable_height_map_ && !has_base_T_camera_) {
    if (tf_buffer_ptr_ &&
        tf_buffer_ptr_->canTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero)) {
      base_T_camera_ =
          tf_buffer_ptr_->lookupTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero);
      has_base_T_camera_ = true;
      cached_camera_frame_pair_ = base_frame_ + "->" + std::string(msg->header.frame_id.c_str());
      RCLCPP_INFO(this->get_logger(), "Cached base_T_camera for %s",
                  cached_camera_frame_pair_.c_str());
    } else {
      return;
    }
  }

  // Decode compressed image to raw and store in raw buffer
  cv::Mat comp(1, static_cast<int>(msg->data.size()), CV_8UC1,
               const_cast<uint8_t*>(msg->data.data()));
  cv::Mat bgr = cv::imdecode(comp, cv::IMREAD_COLOR);
  if (!bgr.data) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Failed to decode compressed image");
    return;
  }

  // Convert BGR to RGB
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  // Create raw Image message
  sensor_msgs::msg::Image raw_msg;
  raw_msg.header = msg->header;
  raw_msg.height = static_cast<uint32_t>(rgb.rows);
  raw_msg.width = static_cast<uint32_t>(rgb.cols);
  raw_msg.encoding = "rgb8";
  raw_msg.is_bigendian = false;
  raw_msg.step = raw_msg.width * 3;
  raw_msg.data.resize(static_cast<size_t>(rgb.rows * rgb.cols * 3));
  std::memcpy(raw_msg.data.data(), rgb.data, raw_msg.data.size());

  // Store in raw buffer
  if (rgb_raw_buffer_) {
    rgb_raw_buffer_->push(std::move(raw_msg));
  }
}

void SemMapNode::on_rgb_raw(const sensor_msgs::msg::Image::SharedPtr msg,
                            const rclcpp::MessageInfo& info) {
  if (rgb_stats_) {
    process_message(*msg, info, *rgb_stats_, "RGB");
  }

  // Cache base->camera once (only needed for BEV fusion / height mapping)
  if (enable_height_map_ && !has_base_T_camera_) {
    if (tf_buffer_ptr_ &&
        tf_buffer_ptr_->canTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero)) {
      base_T_camera_ =
          tf_buffer_ptr_->lookupTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero);
      has_base_T_camera_ = true;
      cached_camera_frame_pair_ = base_frame_ + "->" + std::string(msg->header.frame_id.c_str());
      RCLCPP_INFO(this->get_logger(), "Cached base_T_camera for %s",
                  cached_camera_frame_pair_.c_str());
    } else {
      return;
    }
  }

  // Convert to RGB8 if needed
  sensor_msgs::msg::Image raw_msg;
  if (msg->encoding == "rgb8") {
    // Already RGB8, use as-is
    raw_msg = *msg;
  } else if (msg->encoding == "bgr8") {
    // Convert BGR to RGB
    cv::Mat bgr(msg->height, msg->width, CV_8UC3, const_cast<uint8_t*>(msg->data.data()));
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    raw_msg.header = msg->header;
    raw_msg.height = msg->height;
    raw_msg.width = msg->width;
    raw_msg.encoding = "rgb8";
    raw_msg.is_bigendian = false;
    raw_msg.step = msg->width * 3;
    raw_msg.data.resize(static_cast<size_t>(rgb.rows * rgb.cols * 3));
    std::memcpy(raw_msg.data.data(), rgb.data, raw_msg.data.size());
  } else if (msg->encoding == "bgra8") {
    // Convert BGRA to RGB
    cv::Mat bgra(msg->height, msg->width, CV_8UC4, const_cast<uint8_t*>(msg->data.data()));
    cv::Mat rgb;
    cv::cvtColor(bgra, rgb, cv::COLOR_BGRA2RGB);

    raw_msg.header = msg->header;
    raw_msg.height = msg->height;
    raw_msg.width = msg->width;
    raw_msg.encoding = "rgb8";
    raw_msg.is_bigendian = false;
    raw_msg.step = msg->width * 3;
    raw_msg.data.resize(static_cast<size_t>(rgb.rows * rgb.cols * 3));
    std::memcpy(raw_msg.data.data(), rgb.data, raw_msg.data.size());
  } else {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "Unsupported raw image encoding: %s", msg->encoding.c_str());
    return;
  }

  // Store in unified raw buffer
  if (rgb_raw_buffer_) {
    rgb_raw_buffer_->push(std::move(raw_msg));
  }
}

void SemMapNode::on_pointcloud_single(const sensor_msgs::msg::PointCloud2::SharedPtr msg,
                                      const rclcpp::MessageInfo& info) {
  if (pc_stats_) {
    process_message(*msg, info, *pc_stats_, "PointCloud");
  }

  // Cache pointcloud->base transform
  if (!has_pointcloud_T_base_) {
    if (tf_buffer_ptr_->canTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero)) {
      pointcloud_T_base_ =
          tf_buffer_ptr_->lookupTransform(base_frame_, msg->header.frame_id, tf2::TimePointZero);
      has_pointcloud_T_base_ = true;
      cached_pointcloud_frame_pair_ =
          std::string(msg->header.frame_id.c_str()) + "->" + base_frame_;
      RCLCPP_INFO(this->get_logger(), "Cached pointcloud_T_base for %s",
                  cached_pointcloud_frame_pair_.c_str());
    }
  }

  // Push to buffer for processing
  if (pc_buffer_) {
    pc_buffer_->push(std::move(*msg));
  }
}

void SemMapNode::pointcloud_processing_loop() {
  const auto period =
      std::chrono::duration<double>(1.0 / std::max(1e-3, pointcloud_processing_rate_hz_));

  // Hold a pending message locally to avoid race between front()/pop_front() and push().
  sensor_msgs::msg::PointCloud2 pending_msg;
  bool has_pending = false;
  uint64_t pending_jump_gen = 0;
  auto pending_since = std::chrono::steady_clock::now();

  while (pointcloud_thread_running_ && rclcpp::ok()) {
    auto loop_start = std::chrono::steady_clock::now();

    // Drop a pre-time-jump pending message so we don't sit on it until bag
    // time naturally catches up after a rosbag restart (see semantic loop).
    if (has_pending && pending_jump_gen != time_jump_counter_.load(std::memory_order_relaxed)) {
      RCLCPP_WARN(this->get_logger(),
                  "[pointcloud] Dropping pending pointcloud from before time jump");
      has_pending = false;
    }

    // Try to get a message: either from pending or by popping the buffer
    if (!has_pending) {
      if (pc_buffer_ && has_pointcloud_T_base_ && pc_buffer_->try_pop_front(pending_msg)) {
        has_pending = true;
        pending_jump_gen = time_jump_counter_.load(std::memory_order_relaxed);
        pending_since = std::chrono::steady_clock::now();
      }
    }

    if (has_pending) {
      const auto& msg = pending_msg;
      const uint64_t stamp_ns =
          static_cast<uint64_t>(msg.header.stamp.sec) * 1000000000ULL + msg.header.stamp.nanosec;

      if (last_processed_pc_stamp_ns_ != stamp_ns) {
        // Check pose interpolation status FIRST (before expensive parsing/transformation)
        geometry_msgs::msg::PoseStamped robot_pose;
        PoseInterpolationStatus interp_status = interpolate_pose(stamp_ns, robot_pose);

        if (interp_status == PoseInterpolationStatus::TOO_NEW) {
          // Pointcloud is in the future - keep pending, wait for pose to arrive.
          // Bounded: drop after MAX_PENDING_HOLD_SEC so a stalled pose source or
          // clock mismatch can't block the pipeline forever.
          if (std::chrono::steady_clock::now() - pending_since >
              std::chrono::duration<double>(MAX_PENDING_HOLD_SEC)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "[pointcloud] Pointcloud timestamp (%.3fs) still newer than "
                                 "newest pose after %.1fs, dropping "
                                 "(pose source stalled or clock mismatch?)",
                                 stamp_ns / 1e9, MAX_PENDING_HOLD_SEC);
            pose_interp_fail_pc_.fetch_add(1, std::memory_order_relaxed);
            has_pending = false;
            continue;
          }
          RCLCPP_DEBUG_THROTTLE(
              this->get_logger(), *this->get_clock(), 5000,
              "[pointcloud] Pointcloud timestamp (%.3fs) is in the future, waiting for pose...",
              stamp_ns / 1e9);
          last_processed_pc_stamp_ns_ = 0;
          auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
          if (loop_elapsed < period) {
            std::this_thread::sleep_for(period - loop_elapsed);
          }
          continue;
        } else if (interp_status == PoseInterpolationStatus::TOO_OLD) {
          // Pointcloud is too old - drop it
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                               "[pointcloud] Pointcloud timestamp (%.3fs) is too old, dropping",
                               stamp_ns / 1e9);
          pose_interp_fail_pc_.fetch_add(1, std::memory_order_relaxed);
          has_pending = false;
          continue;
        } else if (interp_status != PoseInterpolationStatus::SUCCESS) {
          // INSUFFICIENT_POSES or other error - keep pending, wait (bounded)
          if (std::chrono::steady_clock::now() - pending_since >
              std::chrono::duration<double>(MAX_PENDING_HOLD_SEC)) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 5000,
                "[pointcloud] Cannot interpolate pose (status=%d) after %.1fs, dropping",
                static_cast<int>(interp_status), MAX_PENDING_HOLD_SEC);
            pose_interp_fail_pc_.fetch_add(1, std::memory_order_relaxed);
            has_pending = false;
            continue;
          }
          RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                "[pointcloud] Cannot interpolate pose (status=%d), waiting...",
                                static_cast<int>(interp_status));
          pose_interp_fail_pc_.fetch_add(1, std::memory_order_relaxed);
          last_processed_pc_stamp_ns_ = 0;
          auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
          if (loop_elapsed < period) {
            std::this_thread::sleep_for(period - loop_elapsed);
          }
          continue;
        }

        // Pose interpolation succeeded, now process pointcloud
        last_processed_pc_stamp_ns_ = stamp_ns;
        auto proc_start = std::chrono::steady_clock::now();
        RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                              "[pointcloud] Processing, buffer_size=%zu", pc_buffer_->size());

        try {
          // Parse pointcloud XYZ (reuse pre-allocated buffer)
          pointcloud_buffer_.clear();
          if (!parse_pointcloud_xyz(msg, pointcloud_buffer_)) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                 "[pointcloud] Failed to parse XYZ");
            has_pending = false;
            continue;
          }

          if (pointcloud_buffer_.empty()) {
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "[pointcloud] No valid points");
            has_pending = false;
            continue;
          }

          size_t n_points = pointcloud_buffer_.size() / 3;
          RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                "[pointcloud] Parsed %zu points", n_points);

          // Transform points to base frame using cached transform (reuse pre-allocated buffer)
          pointcloud_transformed_buffer_.clear();
          pointcloud_transformed_buffer_.reserve(pointcloud_buffer_.size());

          // Extract rotation and translation from cached transform
          const auto& t = pointcloud_T_base_.transform.translation;
          const auto& q = pointcloud_T_base_.transform.rotation;

          // Convert quaternion to rotation matrix
          Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
          Eigen::Matrix3d R = quat.toRotationMatrix();
          Eigen::Vector3d translation(t.x, t.y, t.z);

          // Transform each point
          for (size_t i = 0; i < n_points; ++i) {
            Eigen::Vector3d p(pointcloud_buffer_[i * 3], pointcloud_buffer_[i * 3 + 1],
                              pointcloud_buffer_[i * 3 + 2]);
            Eigen::Vector3d p_base = R * p + translation;
            pointcloud_transformed_buffer_.push_back(static_cast<float>(p_base.x()));
            pointcloud_transformed_buffer_.push_back(static_cast<float>(p_base.y()));
            pointcloud_transformed_buffer_.push_back(static_cast<float>(p_base.z()));
          }

          // robot_pose already populated from status check above

          // Extract rotation and translation (robot pose in world frame)
          Eigen::Vector3d robot_position(robot_pose.pose.position.x, robot_pose.pose.position.y,
                                         robot_pose.pose.position.z);
          Eigen::Quaterniond robot_quat(
              robot_pose.pose.orientation.w, robot_pose.pose.orientation.x,
              robot_pose.pose.orientation.y, robot_pose.pose.orientation.z);
          Eigen::Matrix3d robot_R = robot_quat.toRotationMatrix();

          // Store map center (used by visualization and stats)
          // In legacy mode: update immediately from robot position
          // In multiprocessing mode: map center is ONLY updated from elevation worker via
          // read_bev_map() (elevation worker determines map center when it calls move_to() based on
          // robot position)
          if (!sem_map_->is_multiprocessing_enabled()) {
            // Legacy mode: update map_center_ and move elevation map
            std::array<double, 3> position_array = {robot_position.x(), robot_position.y(),
                                                    robot_position.z()};
            {
              std::lock_guard<std::mutex> lock(map_center_mutex_);
              map_center_ = position_array;
            }

            Eigen::Matrix3d rotation_identity = Eigen::Matrix3d::Identity();
            sem_map_->move_to(robot_position, rotation_identity);
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "[pointcloud] Moved map center to [%.2f, %.2f, %.2f]",
                                  robot_position.x(), robot_position.y(), robot_position.z());
          }
          // In multiprocessing mode: map_center_ is NOT updated here - it comes from elevation
          // worker via read_bev_map()

          // Process pointcloud (multiprocessing or legacy mode)
          bool process_success = false;

          if (sem_map_->is_multiprocessing_enabled()) {
            // Multiprocessing mode: Write to buffer for elevation worker
            std::vector<std::string> channels = {"x", "y", "z"};

            process_success = sem_map_->write_pointcloud_to_buffer(
                pointcloud_transformed_buffer_, channels, stamp_ns, robot_R, robot_position);

            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "[pointcloud][MP] Wrote %zu points to buffer", n_points);
          } else {
            // Legacy mode: Process directly
            std::vector<std::string> channels = {"x", "y", "z"};
            process_success = sem_map_->process_pointcloud(pointcloud_transformed_buffer_, channels,
                                                           robot_R, robot_position);

            if (process_success) {
              RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                    "[pointcloud] Processed %zu points", n_points);
            } else {
              RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                   "[pointcloud] Failed to process pointcloud");
            }
          }

          // Record processing time
          auto proc_elapsed = std::chrono::steady_clock::now() - proc_start;
          thread_stats_["Pointcloud Processing"].record(
              std::chrono::duration<double, std::milli>(proc_elapsed).count());

          has_pending = false;
        } catch (const std::exception& e) {
          std::string error_msg = std::string("[pointcloud] Processing error: ") + e.what();
          {
            std::lock_guard<std::mutex> lock(stats_data_.mutex);
            stats_data_.status_messages.push_back({error_msg, true});
          }
          RCLCPP_ERROR(this->get_logger(), "%s", error_msg.c_str());

          has_pending = false;
        }
      } else {
        // Timestamp already processed, discard
        has_pending = false;
      }
    } else {
      RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                            "[pointcloud] Waiting: buffer_size=%zu, has_transform=%d",
                            pc_buffer_ ? pc_buffer_->size() : 0, has_pointcloud_T_base_);
    }

    // Sleep for remaining time to maintain rate
    auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
    if (loop_elapsed < period) {
      std::this_thread::sleep_for(period - loop_elapsed);
    }
  }
}

void SemMapNode::ros_publishing_loop() {
  const auto period = std::chrono::duration<double>(1.0 / std::max(1e-3, ros_publish_rate_hz_));

  while (ros_publish_thread_running_ && rclcpp::ok()) {
    auto loop_start = std::chrono::steady_clock::now();

    try {
      auto proc_start = std::chrono::steady_clock::now();

      // ====================================================================
      // PART 1: Publish semantic results (multiprocessing mode only)
      // ====================================================================
      if (sem_map_->is_multiprocessing_enabled()) {
        traversability_buffer_.clear();
        segmentation_buffer_.clear();
        confidence_buffer_.clear();
        int trav_h = 0, trav_w = 0;
        double semantic_timestamp = 0.0;

        bool has_semantic = sem_map_->read_semantic_results_from_buffer(
            traversability_buffer_, segmentation_buffer_, confidence_buffer_, trav_h, trav_w,
            semantic_timestamp);

        if (has_semantic) {
          size_t expected_size = static_cast<size_t>(trav_h * trav_w);

          // Build header from semantic result timestamp
          std_msgs::msg::Header sem_header;
          uint64_t sem_stamp_ns = static_cast<uint64_t>(semantic_timestamp * 1e9);
          sem_header.stamp.sec = static_cast<int32_t>(sem_stamp_ns / 1000000000ULL);
          sem_header.stamp.nanosec = static_cast<uint32_t>(sem_stamp_ns % 1000000000ULL);
          sem_header.frame_id = camera_frame_id_.empty() ? world_frame_ : camera_frame_id_;

          // Publish traversability
          if (publish_traversability_ && traversability_pub_) {
            if (traversability_u16_buffer_.size() != expected_size) {
              traversability_u16_buffer_.resize(expected_size);
            }

            const float* src = traversability_buffer_.data();
            uint16_t* dst = traversability_u16_buffer_.data();
            for (size_t i = 0; i < expected_size; ++i) {
              float v = std::clamp(src[i], 0.0f, 1.0f);
              dst[i] = static_cast<uint16_t>(v * 65535.0f + 0.5f);
            }

            sensor_msgs::msg::Image out;
            out.header = sem_header;
            out.height = static_cast<uint32_t>(trav_h);
            out.width = static_cast<uint32_t>(trav_w);
            out.encoding = "mono16";
            out.is_bigendian = false;
            out.step = out.width * 2;
            size_t required_bytes = traversability_u16_buffer_.size() * sizeof(uint16_t);
            if (out.data.size() != required_bytes) {
              out.data.resize(required_bytes);
            }
            std::memcpy(out.data.data(), traversability_u16_buffer_.data(), required_bytes);
            traversability_pub_->publish(out);
          }

          // Publish segmentation
          if (publish_segmentation_ && segmentation_pub_ && segmentation_lut_ready_) {
            if (segmentation_buffer_.size() >= expected_size) {
              segmentation_rgb_buffer_.resize(expected_size * 3);
              const size_t lut_size = segmentation_color_lut_.size();
              const std::array<uint8_t, 3> transparent_color{0, 0, 0};

              for (size_t i = 0; i < expected_size; ++i) {
                float value = segmentation_buffer_[i];
                int label = static_cast<int>(std::round(value));
                bool out_of_range = (label < 0) || (static_cast<size_t>(label) >= lut_size);
                if (out_of_range) {
                  label = 255;
                }

                const auto& color = out_of_range
                                        ? transparent_color
                                        : segmentation_color_lut_[static_cast<size_t>(label)];
                size_t base = i * 3;
                segmentation_rgb_buffer_[base + 0] = color[0];
                segmentation_rgb_buffer_[base + 1] = color[1];
                segmentation_rgb_buffer_[base + 2] = color[2];
              }

              sensor_msgs::msg::Image seg_msg;
              seg_msg.header = sem_header;
              seg_msg.height = static_cast<uint32_t>(trav_h);
              seg_msg.width = static_cast<uint32_t>(trav_w);
              seg_msg.encoding = "rgb8";
              seg_msg.is_bigendian = false;
              seg_msg.step = seg_msg.width * 3;
              seg_msg.data = segmentation_rgb_buffer_;
              segmentation_pub_->publish(seg_msg);
            }
          }

          // Publish confidence
          if (publish_confidence_ && confidence_pub_ && !confidence_buffer_.empty()) {
            size_t conf_size = expected_size;
            if (confidence_buffer_.size() == conf_size) {
              if (confidence_u16_buffer_.size() != conf_size) {
                confidence_u16_buffer_.resize(conf_size);
              }

              const float* src = confidence_buffer_.data();
              uint16_t* dst = confidence_u16_buffer_.data();
              for (size_t i = 0; i < conf_size; ++i) {
                float v = std::clamp(src[i], 0.0f, 1.0f);
                dst[i] = static_cast<uint16_t>(v * 65535.0f + 0.5f);
              }

              sensor_msgs::msg::Image conf_out;
              conf_out.header = sem_header;
              conf_out.height = static_cast<uint32_t>(trav_h);
              conf_out.width = static_cast<uint32_t>(trav_w);
              conf_out.encoding = "mono16";
              conf_out.is_bigendian = false;
              conf_out.step = conf_out.width * 2;
              size_t required_bytes = confidence_u16_buffer_.size() * sizeof(uint16_t);
              if (conf_out.data.size() != required_bytes) {
                conf_out.data.resize(required_bytes);
              }
              std::memcpy(conf_out.data.data(), confidence_u16_buffer_.data(), required_bytes);
              confidence_pub_->publish(conf_out);
            }
          }
        }
      }

      // ====================================================================
      // PART 2: Publish elevation map (only if height mapping enabled)
      // ====================================================================
      if (enable_height_map_) {
        // Retrieve elevation map layers (multiprocessing or legacy mode)
        std::map<std::string, std::vector<float>> layers_data;
        std::vector<std::string> layer_names = {"elevation", "variance", semantic_bev_layer_name_};
        int transformed_h = 0, transformed_w = 0;
        bool has_layers = false;
        uint64_t elevation_stamp_ns = 0;
        std::array<double, 3> map_center = {0.0, 0.0,
                                            0.0};  // Initialize to avoid uninitialized warning

        if (sem_map_->is_multiprocessing_enabled()) {
          // Multiprocessing mode: Read from elevation output buffer (includes map center)
          std::vector<double> map_center_vec;
          has_layers = sem_map_->read_bev_map(layers_data, transformed_h, transformed_w,
                                              elevation_stamp_ns, map_center_vec);
          if (has_layers && map_center_vec.size() == 3) {
            map_center[0] = map_center_vec[0];
            map_center[1] = map_center_vec[1];
            map_center[2] = map_center_vec[2];

            // Mark that we've received at least one valid elevation map result
            has_received_valid_elevation_map_ = true;

            // Update map_center_ member variable for stats publishing (authoritative source in
            // multiprocessing mode)
            {
              std::lock_guard<std::mutex> lock(map_center_mutex_);
              map_center_[0] = map_center[0];
              map_center_[1] = map_center[1];
              map_center_[2] = map_center[2];
            }
          }

          // If no new data, try to republish last valid map
          if (!has_layers) {
            std::lock_guard<std::mutex> lock(last_elevation_map_mutex_);
            if (last_published_elevation_h_ > 0 && last_published_elevation_w_ > 0 &&
                last_published_elevation_layers_.count("elevation") &&
                last_published_elevation_layers_.count("variance") &&
                last_published_elevation_layers_.count(semantic_bev_layer_name_)) {
              // Republish last valid map
              layers_data = last_published_elevation_layers_;
              transformed_h = last_published_elevation_h_;
              transformed_w = last_published_elevation_w_;
              elevation_stamp_ns = last_published_elevation_map_stamp_ns_;
              // Use stored map center from last publication
              map_center[0] = last_published_elevation_map_center_[0];
              map_center[1] = last_published_elevation_map_center_[1];
              map_center[2] = last_published_elevation_map_center_[2];
              has_layers = true;

              // Update map_center_ member variable for stats publishing (when republishing)
              {
                std::lock_guard<std::mutex> mc_lock(map_center_mutex_);
                map_center_ = map_center;
              }
            }
          }
        } else {
          // Legacy mode: read from map_center_ member
          std::lock_guard<std::mutex> lock(map_center_mutex_);
          map_center = map_center_;
          // Legacy mode: Get elevation layers directly (with GPU transformations)
          has_layers = sem_map_->get_elevation_layers_transformed(layer_names, layers_data,
                                                                  transformed_h, transformed_w);
          if (has_layers) {
            // Mark that we've received at least one valid elevation map result
            has_received_valid_elevation_map_ = true;
          }
        }

        // Only publish if we have valid layers AND we've received at least one valid elevation map
        // result
        if (has_layers && has_received_valid_elevation_map_ && transformed_h > 0 &&
            transformed_w > 0 && layers_data.count("elevation") && layers_data.count("variance") &&
            layers_data.count(semantic_bev_layer_name_)) {
          RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                "[ros_publish] Retrieved transformed layers: %dx%d", transformed_w,
                                transformed_h);

          // Create GridMap message
          grid_map_msgs::msg::GridMap grid_map_msg;

          // Set header
          grid_map_msg.header.stamp = this->get_clock()->now();
          grid_map_msg.header.frame_id = world_frame_;

          // Set GridMapInfo
          grid_map_msg.info.resolution = map_resolution_;
          grid_map_msg.info.length_x = map_size_;
          grid_map_msg.info.length_y = map_size_;

          // Set map position (center at map_center position)
          // Note: GridMap is 2D, so z is set to 0.0, but we preserve map_center[2] for TF/pose
          // publishing
          grid_map_msg.info.pose.position.x = map_center[0];
          grid_map_msg.info.pose.position.y = map_center[1];
          grid_map_msg.info.pose.position.z = 0.0;  // GridMap is 2D, z always 0
          grid_map_msg.info.pose.orientation.w = 1.0;
          grid_map_msg.info.pose.orientation.x = 0.0;
          grid_map_msg.info.pose.orientation.y = 0.0;
          grid_map_msg.info.pose.orientation.z = 0.0;

          // Set layers and add data
          if (semantic_bev_layer_name_ == "color_r") {
            // RGB color mode: publish elevation_inpaint + elevation + variance + color_r/g/b
            // color_r: 1.0 = high traversability, 0.0 = low traversability
            // color_g/b: 0.5 = low traversability, 0.0 = high traversability (inverse of color_r,
            // half scale)
            size_t layer_size = static_cast<size_t>(transformed_h * transformed_w);

            // Threshold semantic BEV layer values to 0 or 1
            const auto& semantic_src = layers_data[semantic_bev_layer_name_];
            std::vector<float> semantic_layer_scaled;
            semantic_layer_scaled.reserve(semantic_src.size());
            std::transform(semantic_src.begin(), semantic_src.end(),
                           std::back_inserter(semantic_layer_scaled), [](float v) {
                             return (v >= 0.5f) ? 1.0f : 0.0f;
                           });

            // color_g and color_b: 0.5 where low traversability (color_r == 0), 0.0 where high
            // traversability
            std::vector<float> layer_g(layer_size);
            std::vector<float> layer_b(layer_size);
            std::transform(semantic_layer_scaled.begin(), semantic_layer_scaled.end(),
                           layer_g.begin(), [](float v) {
                             return (1.0f - v) * 0.5f;
                           });
            layer_b = layer_g;

            if (publish_custom_elevation_) {
              // elevation = constant custom value; real elevation published as elevation_real
              std::vector<float> custom_elev_layer(layer_size, custom_elevation_);
              grid_map_msg.layers = {"elevation", "elevation_real", "variance",
                                     "color_r",   "color_g",        "color_b"};
              grid_map_msg.data.push_back(
                  create_float32_multiarray(custom_elev_layer, transformed_h, transformed_w));
              grid_map_msg.data.push_back(create_float32_multiarray(layers_data["elevation"],
                                                                    transformed_h, transformed_w));
            } else {
              grid_map_msg.layers = {"elevation", "variance", "color_r", "color_g", "color_b"};
              grid_map_msg.data.push_back(create_float32_multiarray(layers_data["elevation"],
                                                                    transformed_h, transformed_w));
            }
            grid_map_msg.data.push_back(
                create_float32_multiarray(layers_data["variance"], transformed_h, transformed_w));
            grid_map_msg.data.push_back(
                create_float32_multiarray(semantic_layer_scaled, transformed_h, transformed_w));
            grid_map_msg.data.push_back(
                create_float32_multiarray(layer_g, transformed_h, transformed_w));
            grid_map_msg.data.push_back(
                create_float32_multiarray(layer_b, transformed_h, transformed_w));
          } else {
            // Standard mode: publish elevation + variance + semantic layer
            grid_map_msg.layers = {"elevation", "variance", semantic_bev_layer_name_};
            grid_map_msg.data.push_back(
                create_float32_multiarray(layers_data["elevation"], transformed_h, transformed_w));
            grid_map_msg.data.push_back(
                create_float32_multiarray(layers_data["variance"], transformed_h, transformed_w));
            grid_map_msg.data.push_back(create_float32_multiarray(
                layers_data[semantic_bev_layer_name_], transformed_h, transformed_w));
          }

          // Set indices
          grid_map_msg.outer_start_index = 0;
          grid_map_msg.inner_start_index = 0;

          // Publish GridMap
          if (grid_map_pub_) {
            grid_map_pub_->publish(grid_map_msg);

            // Store last published map for republishing (multiprocessing mode only)
            if (sem_map_->is_multiprocessing_enabled()) {
              last_published_elevation_stamp_ns_ = elevation_stamp_ns;

              // Store a copy of the map data for republishing
              std::lock_guard<std::mutex> lock(last_elevation_map_mutex_);
              last_published_elevation_layers_ = layers_data;
              last_published_elevation_h_ = transformed_h;
              last_published_elevation_w_ = transformed_w;
              last_published_elevation_map_stamp_ns_ = elevation_stamp_ns;
              last_published_elevation_map_center_ = map_center;  // Store map center!
            }

            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "[ros_publish] Published GridMap");
          }

          // Publish map center pose (use actual z coordinate, not GridMap's 0.0)
          if (map_center_pub_) {
            geometry_msgs::msg::PoseStamped map_center_pose;
            map_center_pose.header = grid_map_msg.header;
            map_center_pose.pose.position.x = map_center[0];
            map_center_pose.pose.position.y = map_center[1];
            map_center_pose.pose.position.z = map_center[2];  // Use actual z coordinate
            map_center_pose.pose.orientation.w = 1.0;
            map_center_pose.pose.orientation.x = 0.0;
            map_center_pose.pose.orientation.y = 0.0;
            map_center_pose.pose.orientation.z = 0.0;
            map_center_pub_->publish(map_center_pose);
          }

          // Broadcast TF transform: world_frame -> map_center
          if (tf_broadcaster_) {
            geometry_msgs::msg::TransformStamped transform_msg;
            transform_msg.header = grid_map_msg.header;
            transform_msg.child_frame_id = "map_center";
            transform_msg.transform.translation.x = map_center[0];
            transform_msg.transform.translation.y = map_center[1];
            transform_msg.transform.translation.z = map_center[2];
            transform_msg.transform.rotation.w = 1.0;
            transform_msg.transform.rotation.x = 0.0;
            transform_msg.transform.rotation.y = 0.0;
            transform_msg.transform.rotation.z = 0.0;

            tf_broadcaster_->sendTransform(transform_msg);
            RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                                  "[ros_publish] Broadcasted map_center TF");
          }

        } else {
          RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                                "[ros_publish] Waiting for elevation map layers");
        }
      }  // end if (enable_height_map_)

      // Record processing time for ALL publishing work (semantic + elevation)
      auto proc_elapsed = std::chrono::steady_clock::now() - proc_start;
      thread_stats_["ROS Publishing"].record(
          std::chrono::duration<double, std::milli>(proc_elapsed).count());

    } catch (const std::exception& e) {
      std::string error_msg = std::string("[ros_publish] Error: ") + e.what();
      {
        std::lock_guard<std::mutex> lock(stats_data_.mutex);
        stats_data_.status_messages.push_back({error_msg, true});
      }
      RCLCPP_ERROR(this->get_logger(), "%s", error_msg.c_str());
    }

    // Sleep for remaining time to maintain rate
    auto loop_elapsed = std::chrono::steady_clock::now() - loop_start;
    if (loop_elapsed < period) {
      std::this_thread::sleep_for(period - loop_elapsed);
    }
  }
}

}  // namespace sem_map

RCLCPP_COMPONENTS_REGISTER_NODE(sem_map::SemMapNode)
