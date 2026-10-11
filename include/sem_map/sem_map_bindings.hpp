// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>

namespace sem_map {

/**
 * @brief Wrapper for Python-based unified semantic mapping (SemMap class)
 *
 * Uses pybind11 for C++/Python interoperability.
 * Wraps the Python SemMap class which combines SemanticPipeline and ElevationMapping.
 *
 * Python interpreter lifecycle:
 *  - Initialization: py::initialize_interpreter(false) called once if not initialized
 *  - GIL management: py::gil_scoped_acquire RAII for all Python operations
 *  - Cleanup: Never calls py::finalize_interpreter() to avoid double-free in ROS2 components
 */
class SemMapWrapper {
 public:
  SemMapWrapper();
  ~SemMapWrapper();

  void initialize(
      const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr& node_parameter_interface);

  // Process RGB frame through semantic pipeline
  void process_frame(const uint8_t* data, int height, int width, int channels,
                     const std::string& sensor_name, uint64_t stamp_ns,
                     const Eigen::Matrix3d* pose_R = nullptr,
                     const Eigen::Vector3d* pose_t = nullptr);

  // Retrieve last traversability map as float32 HxW in [0,1]. Returns false if unavailable.
  bool get_last_traversability(std::vector<float>& out_data, int& height, int& width);

  // Retrieve last traversability + segmentation maps (legacy mode)
  bool get_last_semantic_results(std::vector<float>& trav_out, std::vector<float>& seg_out,
                                 int& height, int& width);

  // Set camera intrinsics for image projection
  void set_camera_info(const Eigen::Matrix3d& K, const Eigen::VectorXd& D,
                       const std::string& distortion_model, int width, int height);

  // Process pointcloud and update elevation map
  bool process_pointcloud(const std::vector<float>& points,
                          const std::vector<std::string>& channels, const Eigen::Matrix3d& R,
                          const Eigen::Vector3d& t);

  // Move elevation map center to robot position
  bool move_to(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation);

  // Get specific layer from elevation map
  bool get_elevation_map_layer(const std::string& layer_name, std::vector<float>& output_data,
                               int& height, int& width);

  // Get multiple elevation map layers with transformations applied (rotation + flip for GridMap
  // format) This is much faster than getting layers individually and applying transformations in
  // C++
  bool get_elevation_layers_transformed(const std::vector<std::string>& layer_names,
                                        std::map<std::string, std::vector<float>>& layers_data,
                                        int& height, int& width);

  // ========== Multiprocessing Data Pipeline Methods ==========
  // These methods write/read to CUDA IPC buffers instead of calling Python directly
  // Enabled when use_multiprocessing=True parameter is set

  // Write RGB image to input buffer (non-blocking)
  bool write_rgb_to_buffer(const uint8_t* data, int height, int width, int channels,
                           const std::string& sensor_name, uint64_t stamp_ns,
                           const Eigen::Matrix3d& pose_R, const Eigen::Vector3d& pose_t,
                           const Eigen::Matrix3d& K);

  // Write pointcloud to input buffer (non-blocking)
  bool write_pointcloud_to_buffer(const std::vector<float>& points,
                                  const std::vector<std::string>& channels, uint64_t stamp_ns,
                                  const Eigen::Matrix3d& pose_R, const Eigen::Vector3d& pose_t);

  // Read BEV map layers and map center from output buffer (non-blocking, returns latest)
  bool read_bev_map(std::map<std::string, std::vector<float>>& layers_data, int& height, int& width,
                    uint64_t& stamp_ns, std::vector<double>& map_center);

  // Read semantic results (traversability + segmentation + confidence) from output buffer
  // timestamp is the source frame's sensor time in seconds (double: float32 cannot
  // represent epoch-scale stamps without ~100s quantization)
  bool read_semantic_results_from_buffer(std::vector<float>& trav_out, std::vector<float>& seg_out,
                                         std::vector<float>& conf_out, int& height, int& width,
                                         double& timestamp);

  // Load segmentation color LUT from model config
  bool get_segmentation_color_lut(std::vector<std::array<uint8_t, 3>>& lut);

  // Check if multiprocessing mode is enabled
  bool is_multiprocessing_enabled() const;

  // Signal the elevation worker (multiprocessing mode) to clear all map layers
  bool reset_elevation_map();

  // Read string fields off the Python Parameter object (e.g. for diagnostics).
  // Returns empty string if unavailable.
  std::string get_param_string(const std::string& name);

  void close();

 private:
  void setParameters(
      const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr& node_parameter_interface);

  // Forward declaration to avoid including Python headers in header
  struct PyObjects;
  std::unique_ptr<PyObjects> py_objects_;
};

}  // namespace sem_map
