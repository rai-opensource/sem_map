// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#include "sem_map/sem_map_bindings.hpp"

#include <Python.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>

#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <array>
#include <chrono>
#include <cstdio>
#include <rclcpp/rclcpp.hpp>

namespace py = pybind11;

namespace sem_map {

/**
 * @brief Python objects behind SemMapWrapper, kept out of the header.
 *
 * The GIL must be held whenever these are touched.
 */
struct SemMapWrapper::PyObjects {
  py::object sem_map_;                // SemMap instance
  py::object param_;                  // Parameter instance
  py::object data_pipeline_manager_;  // DataPipelineManager (multiprocessing mode)
  bool use_multiprocessing_ = false;  // Flag for multiprocessing mode
};

SemMapWrapper::SemMapWrapper() : py_objects_(std::make_unique<PyObjects>()) {}

SemMapWrapper::~SemMapWrapper() = default;

void SemMapWrapper::initialize(
    const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr& node_parameter_interface) {
  // Initialize the Python interpreter if not already initialized
  // Safe for ROS2 components: never finalize, let process exit naturally
  // This prevents double-free issues when components are loaded/unloaded
  if (!Py_IsInitialized()) {
    py::initialize_interpreter(false);  // false = don't register atexit handler

    // CRITICAL: For multithreaded applications, we must release the GIL after initialization
    // so other threads can acquire it. pybind11 initialization leaves the GIL held by default.
    PyEval_SaveThread();  // Release GIL, save thread state
    std::fprintf(
        stderr, "[sem_map][bindings] Python interpreter initialized, GIL released for threading\n");
  }

  // Acquire GIL for thread-safe Python operations (pybind11 RAII guard)
  py::gil_scoped_acquire gil;

  try {
    // Add the sem_map path to sys.path
    auto sys = py::module_::import("sys");
    auto path = sys.attr("path");
    std::string module_path = ament_index_cpp::get_package_share_directory("sem_map");
    module_path += "/scripts";
    path.attr("insert")(0, module_path);
    // Import Python modules
    auto parameter = py::module_::import("sem_map.parameter");
    auto sem_map_module = py::module_::import("sem_map.sem_map");

    // Pre-initialize pybind11's lazy numpy C-API bindings while still
    // single-threaded. npy_api::get() (first py::array_t use) hides a
    // function-local static whose initializer imports numpy; if two pipeline
    // threads race it, one holds the GIL and blocks on the static's
    // __cxa_guard while the other holds the guard and blocks re-acquiring
    // the GIL inside the import -> permanent deadlock of all Python callers.
    py::module_::import("numpy");
    {
      py::array_t<uint8_t> warm_u8(1);
      py::array_t<float> warm_f32(1);
      py::array_t<double> warm_f64(1);
      (void)warm_u8;
      (void)warm_f32;
      (void)warm_f64;
    }

    // Create Parameter instance
    py_objects_->param_ = parameter.attr("Parameter")();

    // Set parameters from ROS
    setParameters(node_parameter_interface);

    // Check if multiprocessing mode is enabled
    py_objects_->use_multiprocessing_ =
        py_objects_->param_.attr("use_multiprocessing").cast<bool>();

    if (py_objects_->use_multiprocessing_) {
      // Multiprocessing mode: Create DataPipelineManager
      std::fprintf(stderr, "[sem_map][bindings] Initializing in MULTIPROCESSING mode\n");
      auto data_pipeline_module = py::module_::import("sem_map.data_pipeline");

      // Create config dict from parameters
      py::dict config;
      config["image_height"] = py_objects_->param_.attr("image_height").cast<int>();
      config["image_width"] = py_objects_->param_.attr("image_width").cast<int>();
      config["trav_height"] = py_objects_->param_.attr("trav_height").cast<int>();
      config["trav_width"] = py_objects_->param_.attr("trav_width").cast<int>();
      config["pc_max_points"] = py_objects_->param_.attr("pc_max_points").cast<int>();
      config["buffer_size"] = py_objects_->param_.attr("buffer_size").cast<int>();
      config["map_cells"] = static_cast<int>(py_objects_->param_.attr("map_size").cast<float>() /
                                             py_objects_->param_.attr("resolution").cast<float>());
      config["enable_semantic_pipeline"] = py_objects_->param_.attr("enable_semantic_pipeline");
      config["enable_height_map"] = py_objects_->param_.attr("enable_height_map");

      // Create DataPipelineManager (pass both config and param)
      py_objects_->data_pipeline_manager_ =
          data_pipeline_module.attr("DataPipelineManager")(config, py_objects_->param_);
      py_objects_->data_pipeline_manager_.attr("start")();

      std::fprintf(stderr, "[sem_map][bindings] DataPipelineManager started successfully\n");
    } else {
      // Legacy mode: Create SemMap instance (unified semantic + elevation mapping)
      std::fprintf(stderr,
                   "[sem_map][bindings] Initializing in LEGACY (direct Python call) mode\n");
      py_objects_->sem_map_ = sem_map_module.attr("SemMap")(py_objects_->param_);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] Exception during initialize(): %s\n", e.what());
    throw std::runtime_error(std::string("Python initialization failed: ") + e.what());
  }
  // GIL automatically released when 'gil' goes out of scope
}

void SemMapWrapper::close() {
  try {
    // Acquire GIL for Python operations
    py::gil_scoped_acquire gil;

    // Shutdown DataPipelineManager if in multiprocessing mode
    if (py_objects_->use_multiprocessing_ && !py_objects_->data_pipeline_manager_.is_none()) {
      // Release GIL during long-running shutdown operation
      {
        py::gil_scoped_release gil_release;
        {
          py::gil_scoped_acquire gil_reacquire;
          py_objects_->data_pipeline_manager_.attr("shutdown")();
        }
      }
    }

    // Close SemMap if in legacy mode
    if (!py_objects_->use_multiprocessing_ && !py_objects_->sem_map_.is_none()) {
      py_objects_->sem_map_.attr("close")();
    }

    // Drop references to allow Python GC to clean up
    // Note: We don't call py::finalize_interpreter() - Python interpreter stays alive
    // This is correct behavior for ROS2 components
    try {
      py_objects_->sem_map_ = py::object();
    } catch (...) {
    }
    try {
      py_objects_->data_pipeline_manager_ = py::object();
    } catch (...) {
    }
    try {
      py_objects_->param_ = py::object();
    } catch (...) {
    }

  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] close error: %s\n", e.what());
  }
}

void SemMapWrapper::setParameters(
    const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr& node_parameter_interface) {
  // Acquire GIL for thread-safe Python operations
  py::gil_scoped_acquire gil;

  // Get all parameters names and types from Python
  py::list paramNames = py_objects_->param_.attr("get_names")();
  py::list paramTypes = py_objects_->param_.attr("get_types")();

  // Try to find each parameter in the ROS parameter server
  const int n = py::len(paramNames);
  for (int i = 0; i < n; ++i) {
    std::string type = paramTypes[i].cast<std::string>();
    std::string name = paramNames[i].cast<std::string>();

    rclcpp::Parameter param;
    bool found = node_parameter_interface->get_parameter(name, param);
    if (!found) {
      // Also check under the semantic_pipeline.* namespace commonly used in YAML
      std::string namespaced = std::string("semantic_pipeline.") + name;
      found = node_parameter_interface->get_parameter(namespaced, param);
    }
    if (found) {
      if (type == "float") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
          py_objects_->param_.attr("set_value")(name, static_cast<float>(param.as_int()));
        } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE) {
          py_objects_->param_.attr("set_value")(name, static_cast<float>(param.as_double()));
        }
      } else if (type == "str") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
          py_objects_->param_.attr("set_value")(name, param.as_string());
        }
      } else if (type == "bool") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) {
          py_objects_->param_.attr("set_value")(name, param.as_bool());
        }
      } else if (type == "int") {
        if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER) {
          py_objects_->param_.attr("set_value")(name, param.as_int());
        }
      }
    }
  }

  // Update parameters (validation, derived values, etc.)
  py_objects_->param_.attr("update")();
}

void SemMapWrapper::process_frame(const uint8_t* data, int height, int width, int channels,
                                  const std::string& sensor_name, uint64_t stamp_ns,
                                  const Eigen::Matrix3d* pose_R, const Eigen::Vector3d* pose_t) {
  // Validate input
  if (!data) {
    throw std::runtime_error("Null data pointer in process_frame");
  }

  {
    // Acquire GIL for thread-safe Python operations
    py::gil_scoped_acquire gil;

    // Create Python bytes object with explicit copy from C++ buffer
    const size_t size_bytes =
        static_cast<size_t>(height) * static_cast<size_t>(width) * static_cast<size_t>(channels);

    // Use PyBytes_FromStringAndSize to force immediate deep copy
    // This ensures data is copied while we still hold the message copy
    PyObject* bytes_obj = PyBytes_FromStringAndSize(reinterpret_cast<const char*>(data),
                                                    static_cast<Py_ssize_t>(size_bytes));
    if (!bytes_obj) {
      throw std::runtime_error("Failed to create Python bytes object");
    }
    py::bytes buf = py::reinterpret_steal<py::bytes>(bytes_obj);

    // Call process_frame with or without pose
    if (pose_R && pose_t) {
      // pybind11/eigen.h automatically converts Eigen types to NumPy arrays
      py_objects_->sem_map_.attr("process_frame")(buf, height, width, channels, sensor_name,
                                                  py::int_(stamp_ns), *pose_R, *pose_t);
    } else {
      // Call without pose (no fusion)
      py_objects_->sem_map_.attr("process_frame")(buf, height, width, channels, sensor_name,
                                                  py::int_(stamp_ns));
    }
  }  // GIL released here
}

bool SemMapWrapper::get_last_traversability(std::vector<float>& out_data, int& height, int& width) {
  py::gil_scoped_acquire gil;
  try {
    // Call get_last_traversability() method which handles tensor conversion
    py::object result = py_objects_->sem_map_.attr("get_last_traversability")();
    if (result.is_none()) {
      return false;
    }

    py::array arr = py::reinterpret_borrow<py::array>(result);
    if (!arr || arr.ndim() != 2) {
      return false;
    }
    // Ensure contiguous float32 array
    py::array_t<float, py::array::c_style | py::array::forcecast> arr_f =
        arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    height = static_cast<int>(arr_f.shape(0));
    width = static_cast<int>(arr_f.shape(1));
    out_data.resize(static_cast<size_t>(height * width));
    std::memcpy(out_data.data(), arr_f.data(), static_cast<size_t>(height * width) * sizeof(float));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] get_last_traversability error: %s\n", e.what());
    return false;
  }
}

bool SemMapWrapper::get_last_semantic_results(std::vector<float>& trav_out,
                                              std::vector<float>& seg_out, int& height,
                                              int& width) {
  py::gil_scoped_acquire gil;
  try {
    if (py_objects_->sem_map_.is_none()) {
      return false;
    }
    py::object result = py_objects_->sem_map_.attr("get_last_semantic_results")();
    if (result.is_none()) {
      return false;
    }

    py::tuple result_tuple = py::reinterpret_borrow<py::tuple>(result);
    if (py::len(result_tuple) != 2) {
      std::fprintf(stderr,
                   "[sem_map][bindings] get_last_semantic_results: expected tuple of length 2\n");
      return false;
    }

    py::array trav_arr = py::reinterpret_borrow<py::array>(result_tuple[0]);
    py::array seg_arr = py::reinterpret_borrow<py::array>(result_tuple[1]);

    py::array_t<float, py::array::c_style | py::array::forcecast> trav_np =
        trav_arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    py::array_t<float, py::array::c_style | py::array::forcecast> seg_np =
        seg_arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

    height = static_cast<int>(trav_np.shape(0));
    width = static_cast<int>(trav_np.shape(1));

    size_t size = static_cast<size_t>(height * width);
    trav_out.resize(size);
    seg_out.resize(size);
    std::memcpy(trav_out.data(), trav_np.data(), size * sizeof(float));
    std::memcpy(seg_out.data(), seg_np.data(), size * sizeof(float));

    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] get_last_semantic_results error: %s\n", e.what());
    return false;
  }
}

bool SemMapWrapper::get_segmentation_color_lut(std::vector<std::array<uint8_t, 3>>& lut) {
  py::gil_scoped_acquire gil;
  try {
    auto models_module = py::module_::import("sem_map.perception.models");
    std::string backbone = py_objects_->param_.attr("backbone_name").cast<std::string>();
    std::string seg_layer = py_objects_->param_.attr("seg_layer").cast<std::string>();

    py::object categories_obj = models_module.attr("get_categories")(backbone, seg_layer);
    py::list categories = py::reinterpret_borrow<py::list>(categories_obj);
    if (categories.empty()) {
      std::fprintf(stderr, "[sem_map][bindings] get_segmentation_color_lut: categories empty\n");
      return false;
    }

    int max_id = 0;
    for (auto item : categories) {
      py::dict cat = py::reinterpret_borrow<py::dict>(item);
      int class_id = cat["id"].cast<int>();
      max_id = std::max(max_id, class_id);
    }

    lut.assign(static_cast<size_t>(max_id + 1), std::array<uint8_t, 3>{0, 0, 0});

    for (auto item : categories) {
      py::dict cat = py::reinterpret_borrow<py::dict>(item);
      int class_id = cat["id"].cast<int>();
      if (!cat.contains("color")) {
        continue;
      }
      py::list color_list = py::reinterpret_borrow<py::list>(cat["color"]);
      if (py::len(color_list) != 3) {
        continue;
      }
      std::array<uint8_t, 3> color{};
      for (size_t i = 0; i < 3; ++i) {
        color[i] = static_cast<uint8_t>(std::clamp(color_list[i].cast<int>(), 0, 255));
      }
      if (class_id >= 0 && static_cast<size_t>(class_id) < lut.size()) {
        lut[static_cast<size_t>(class_id)] = color;
      }
    }

    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] get_segmentation_color_lut error: %s\n", e.what());
    return false;
  }
}

void SemMapWrapper::set_camera_info(const Eigen::Matrix3d& K, const Eigen::VectorXd& D,
                                    const std::string& distortion_model, int width, int height) {
  py::gil_scoped_acquire gil;
  try {
    // pybind11/eigen.h automatically converts Eigen types to NumPy arrays
    py_objects_->sem_map_.attr("set_camera_info")(K, D, distortion_model, width, height);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] set_camera_info error: %s\n", e.what());
  }
}

bool SemMapWrapper::process_pointcloud(const std::vector<float>& points,
                                       const std::vector<std::string>& channels,
                                       const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  py::gil_scoped_acquire gil;
  try {
    // Convert points vector to numpy array (N, 3)
    size_t n_points = points.size() / 3;
    std::vector<size_t> points_shape = {n_points, 3};
    py::array_t<float> points_arr(points_shape, points.data());

    // Convert channels to Python list
    py::list channels_list;
    for (const auto& ch : channels) {
      channels_list.append(ch);
    }

    // pybind11/eigen.h automatically converts Eigen types to NumPy arrays
    py::object result =
        py_objects_->sem_map_.attr("process_pointcloud")(points_arr, channels_list, R, t);

    // Convert Python bool to C++ bool
    if (!result.is_none() && py::isinstance<py::bool_>(result)) {
      return result.cast<bool>();
    }

    // If no return value or not a bool, assume success if no exception
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] process_pointcloud error: %s\n", e.what());
    return false;
  }
}

bool SemMapWrapper::move_to(const Eigen::Vector3d& position, const Eigen::Matrix3d& rotation) {
  py::gil_scoped_acquire gil;
  try {
    // pybind11/eigen.h automatically converts Eigen types to NumPy arrays
    py_objects_->sem_map_.attr("move_to")(position, rotation);
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] move_to error: %s\n", e.what());
    return false;
  }
}

bool SemMapWrapper::get_elevation_map_layer(const std::string& layer_name,
                                            std::vector<float>& output_data, int& height,
                                            int& width) {
  py::gil_scoped_acquire gil;
  try {
    // Call Python method
    py::object result = py_objects_->sem_map_.attr("get_elevation_map_layer")(layer_name);
    if (result.is_none()) {
      return false;
    }

    py::array arr = py::reinterpret_borrow<py::array>(result);
    if (!arr || arr.ndim() != 2) {
      return false;
    }

    // Ensure contiguous float32 array
    py::array_t<float, py::array::c_style | py::array::forcecast> arr_f =
        arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    height = static_cast<int>(arr_f.shape(0));
    width = static_cast<int>(arr_f.shape(1));
    output_data.resize(static_cast<size_t>(height * width));
    std::memcpy(output_data.data(), arr_f.data(),
                static_cast<size_t>(height * width) * sizeof(float));
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] get_elevation_map_layer error: %s\n", e.what());
    return false;
  }
}

bool SemMapWrapper::get_elevation_layers_transformed(
    const std::vector<std::string>& layer_names,
    std::map<std::string, std::vector<float>>& layers_data, int& height, int& width) {
  py::gil_scoped_acquire gil;
  try {
    // Convert std::vector to Python list
    py::list py_layer_names;
    for (const auto& name : layer_names) {
      py_layer_names.append(py::str(name));
    }

    // Call Python method that returns dict of transformed layers
    py::object result =
        py_objects_->sem_map_.attr("get_elevation_layers_transformed")(py_layer_names);
    if (result.is_none()) {
      return false;
    }

    // Result should be a dictionary
    py::dict layers_dict = py::reinterpret_borrow<py::dict>(result);
    if (layers_dict.empty()) {
      return false;
    }

    // Extract dimensions from first layer (all should have same dimensions)
    bool dimensions_set = false;

    // Iterate over dictionary entries
    for (auto item : layers_dict) {
      std::string layer_name = py::str(item.first);
      py::array arr = py::reinterpret_borrow<py::array>(item.second);

      if (!arr || arr.ndim() != 2) {
        std::fprintf(stderr, "[sem_map][bindings] Invalid array for layer %s\n",
                     layer_name.c_str());
        continue;
      }

      // Ensure contiguous float32 array
      py::array_t<float, py::array::c_style | py::array::forcecast> arr_f =
          arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

      // Set dimensions from first valid layer
      if (!dimensions_set) {
        height = static_cast<int>(arr_f.shape(0));
        width = static_cast<int>(arr_f.shape(1));
        dimensions_set = true;
      }

      // Copy data to output map
      size_t data_size = static_cast<size_t>(height * width);
      layers_data[layer_name].resize(data_size);
      std::memcpy(layers_data[layer_name].data(), arr_f.data(), data_size * sizeof(float));
    }

    return dimensions_set && !layers_data.empty();

  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] get_elevation_layers_transformed error: %s\n",
                 e.what());
    return false;
  }
}

// ========== Multiprocessing Data Pipeline Methods ==========

bool SemMapWrapper::write_rgb_to_buffer(const uint8_t* data, int height, int width, int channels,
                                        const std::string& sensor_name, uint64_t stamp_ns,
                                        const Eigen::Matrix3d& pose_R,
                                        const Eigen::Vector3d& pose_t, const Eigen::Matrix3d& K) {
  if (!py_objects_->use_multiprocessing_) {
    std::fprintf(stderr, "[sem_map][bindings] write_rgb_to_buffer: multiprocessing not enabled\n");
    return false;
  }

  if (!data) {
    std::fprintf(stderr, "[sem_map][bindings] write_rgb_to_buffer: null data pointer\n");
    return false;
  }

  // GIL timing instrumentation
  auto gil_wait_start = std::chrono::steady_clock::now();

  // Acquire GIL for Python call
  py::gil_scoped_acquire gil;

  auto gil_acquired = std::chrono::steady_clock::now();
  auto gil_wait_us =
      std::chrono::duration_cast<std::chrono::microseconds>(gil_acquired - gil_wait_start).count();

  try {
    // Create numpy array from C++ data (zero-copy view)
    py::array_t<uint8_t> img_array({height, width, channels}, data);

    // Call write_rgb method
    // Note: pybind11/eigen.h automatically converts Eigen types to NumPy arrays
    py_objects_->data_pipeline_manager_.attr("write_rgb")(img_array, sensor_name, stamp_ns, pose_R,
                                                          pose_t, K);

    auto gil_released = std::chrono::steady_clock::now();
    auto gil_hold_us =
        std::chrono::duration_cast<std::chrono::microseconds>(gil_released - gil_acquired).count();
    auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(gil_released - gil_wait_start)
            .count();

    // Log timing (throttled to avoid spam)
    static int call_count = 0;
    if (++call_count % 100 == 0) {
      std::fprintf(stderr, "[sem_map][GIL] write_rgb: wait=%ldμs hold=%ldμs total=%ldμs (n=%d)\n",
                   gil_wait_us, gil_hold_us, total_us, call_count);
    }

    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] write_rgb_to_buffer error: %s\n", e.what());
    return false;
  }
}

bool SemMapWrapper::write_pointcloud_to_buffer(const std::vector<float>& points,
                                               const std::vector<std::string>& channels,
                                               uint64_t stamp_ns, const Eigen::Matrix3d& pose_R,
                                               const Eigen::Vector3d& pose_t) {
  if (!py_objects_->use_multiprocessing_) {
    std::fprintf(stderr,
                 "[sem_map][bindings] write_pointcloud_to_buffer: multiprocessing not enabled\n");
    return false;
  }

  if (points.empty()) {
    std::fprintf(stderr, "[sem_map][bindings] write_pointcloud_to_buffer: empty pointcloud\n");
    return false;
  }

  // GIL timing instrumentation
  auto gil_wait_start = std::chrono::steady_clock::now();

  // Acquire GIL for Python call
  py::gil_scoped_acquire gil;

  auto gil_acquired = std::chrono::steady_clock::now();
  auto gil_wait_us =
      std::chrono::duration_cast<std::chrono::microseconds>(gil_acquired - gil_wait_start).count();

  try {
    // Convert points vector to numpy array
    // Assume points are in shape (N, C) where C = number of channels
    int num_channels = static_cast<int>(channels.size());
    int num_points = static_cast<int>(points.size()) / num_channels;

    py::array_t<float> points_array({num_points, num_channels});
    std::memcpy(points_array.mutable_data(), points.data(), points.size() * sizeof(float));

    // Convert channels to Python list
    py::list channels_list;
    for (const auto& ch : channels) {
      channels_list.append(ch);
    }

    // Call write_pointcloud method
    // Note: pybind11/eigen.h automatically converts Eigen::Matrix3d and Eigen::Vector3d to NumPy
    // arrays
    py_objects_->data_pipeline_manager_.attr("write_pointcloud")(points_array, channels_list,
                                                                 stamp_ns, pose_R, pose_t);

    auto gil_released = std::chrono::steady_clock::now();
    auto gil_hold_us =
        std::chrono::duration_cast<std::chrono::microseconds>(gil_released - gil_acquired).count();
    auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(gil_released - gil_wait_start)
            .count();

    // Log timing (throttled to avoid spam)
    static int call_count = 0;
    if (++call_count % 100 == 0) {
      std::fprintf(stderr,
                   "[sem_map][GIL] write_pointcloud: wait=%ldμs hold=%ldμs total=%ldμs (n=%d)\n",
                   gil_wait_us, gil_hold_us, total_us, call_count);
    }

    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] write_pointcloud_to_buffer error: %s\n", e.what());
    return false;
  }
}

std::string SemMapWrapper::get_param_string(const std::string& name) {
  py::gil_scoped_acquire gil;
  try {
    if (!py_objects_->param_) {
      return "";
    }
    return py_objects_->param_.attr(name.c_str()).cast<std::string>();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] get_param_string(%s) error: %s\n", name.c_str(),
                 e.what());
    return "";
  }
}

bool SemMapWrapper::reset_elevation_map() {
  if (!py_objects_->use_multiprocessing_) {
    std::fprintf(stderr, "[sem_map][bindings] reset_elevation_map: multiprocessing not enabled\n");
    return false;
  }

  py::gil_scoped_acquire gil;
  try {
    py_objects_->data_pipeline_manager_.attr("reset_elevation_map")();
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] reset_elevation_map error: %s\n", e.what());
    return false;
  }
}

bool SemMapWrapper::read_bev_map(std::map<std::string, std::vector<float>>& layers_data,
                                 int& height, int& width, uint64_t& stamp_ns,
                                 std::vector<double>& map_center) {
  if (!py_objects_->use_multiprocessing_) {
    std::fprintf(stderr, "[sem_map][bindings] read_bev_map: multiprocessing not enabled\n");
    return false;
  }

  // GIL timing instrumentation
  auto gil_wait_start = std::chrono::steady_clock::now();

  // Acquire GIL for Python call
  py::gil_scoped_acquire gil;

  auto gil_acquired = std::chrono::steady_clock::now();
  auto gil_wait_us =
      std::chrono::duration_cast<std::chrono::microseconds>(gil_acquired - gil_wait_start).count();

  try {
    // Call read_bev_map method (returns tuple or None)
    py::object result = py_objects_->data_pipeline_manager_.attr("read_bev_map")();

    auto gil_released = std::chrono::steady_clock::now();
    auto gil_hold_us =
        std::chrono::duration_cast<std::chrono::microseconds>(gil_released - gil_acquired).count();
    auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(gil_released - gil_wait_start)
            .count();

    if (result.is_none()) {
      // No data available - still log timing
      static int empty_count = 0;
      if (++empty_count % 1000 == 0) {
        std::fprintf(
            stderr,
            "[sem_map][GIL] read_bev_map (empty): wait=%ldμs hold=%ldμs total=%ldμs (n=%d)\n",
            gil_wait_us, gil_hold_us, total_us, empty_count);
      }
      return false;
    }

    // Result should be a tuple: (layers_dict, map_center, timestamp_ns)
    py::tuple result_tuple = py::reinterpret_borrow<py::tuple>(result);
    if (py::len(result_tuple) != 3) {
      std::fprintf(
          stderr,
          "[sem_map][bindings] read_bev_map: invalid result format, expected tuple of length 3\n");
      return false;
    }

    py::dict layers_dict = py::reinterpret_borrow<py::dict>(result_tuple[0]);
    py::tuple map_center_tuple = py::reinterpret_borrow<py::tuple>(result_tuple[1]);
    stamp_ns = result_tuple[2].cast<uint64_t>();

    // Extract map center (required - should always be present)
    if (py::len(map_center_tuple) != 3) {
      std::fprintf(stderr,
                   "[sem_map][bindings] read_bev_map: invalid map_center format, expected tuple of "
                   "length 3\n");
      return false;
    }
    map_center.resize(3);
    map_center[0] = map_center_tuple[0].cast<double>();
    map_center[1] = map_center_tuple[1].cast<double>();
    map_center[2] = map_center_tuple[2].cast<double>();

    if (layers_dict.empty()) {
      return false;
    }

    // Extract dimensions and data from each layer
    bool dimensions_set = false;

    for (auto item : layers_dict) {
      std::string layer_name = py::str(item.first);
      py::array arr = py::reinterpret_borrow<py::array>(item.second);

      if (!arr || arr.ndim() != 2) {
        std::fprintf(stderr, "[sem_map][bindings] Invalid array for layer %s\n",
                     layer_name.c_str());
        continue;
      }

      // Ensure contiguous float32 array
      py::array_t<float, py::array::c_style | py::array::forcecast> arr_f =
          arr.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();

      // Set dimensions from first valid layer
      if (!dimensions_set) {
        height = static_cast<int>(arr_f.shape(0));
        width = static_cast<int>(arr_f.shape(1));
        dimensions_set = true;
      }

      // Copy data to output map
      size_t data_size = static_cast<size_t>(height * width);
      layers_data[layer_name].resize(data_size);
      std::memcpy(layers_data[layer_name].data(), arr_f.data(), data_size * sizeof(float));
    }

    // Log timing for successful reads (throttled)
    static int success_count = 0;
    if (++success_count % 100 == 0) {
      std::fprintf(
          stderr,
          "[sem_map][GIL] read_bev_map (success): wait=%ldμs hold=%ldμs total=%ldμs (n=%d)\n",
          gil_wait_us, gil_hold_us, total_us, success_count);
    }

    return dimensions_set && !layers_data.empty();

  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] read_bev_map error: %s\n", e.what());
    return false;
  }
}

bool SemMapWrapper::read_semantic_results_from_buffer(std::vector<float>& trav_out,
                                                      std::vector<float>& seg_out,
                                                      std::vector<float>& conf_out, int& height,
                                                      int& width, double& timestamp) {
  if (!py_objects_->use_multiprocessing_) {
    std::fprintf(
        stderr,
        "[sem_map][bindings] read_semantic_results_from_buffer: multiprocessing not enabled\n");
    return false;
  }

  auto gil_wait_start = std::chrono::steady_clock::now();
  py::gil_scoped_acquire gil;
  auto gil_acquired = std::chrono::steady_clock::now();
  auto gil_wait_us =
      std::chrono::duration_cast<std::chrono::microseconds>(gil_acquired - gil_wait_start).count();

  try {
    py::object result = py_objects_->data_pipeline_manager_.attr("read_semantic_results")();

    auto gil_released = std::chrono::steady_clock::now();
    auto gil_hold_us =
        std::chrono::duration_cast<std::chrono::microseconds>(gil_released - gil_acquired).count();
    auto total_us =
        std::chrono::duration_cast<std::chrono::microseconds>(gil_released - gil_wait_start)
            .count();

    if (result.is_none()) {
      static int empty_count = 0;
      if (++empty_count % 1000 == 0) {
        std::fprintf(stderr,
                     "[sem_map][GIL] read_semantic_results (empty): wait=%ldμs hold=%ldμs "
                     "total=%ldμs (n=%d)\n",
                     gil_wait_us, gil_hold_us, total_us, empty_count);
      }
      return false;
    }

    py::tuple result_tuple = py::reinterpret_borrow<py::tuple>(result);
    if (py::len(result_tuple) != 6) {
      std::fprintf(
          stderr,
          "[sem_map][bindings] read_semantic_results: expected tuple of length 6, got %ld\n",
          py::len(result_tuple));
      return false;
    }

    py::object trav_tensor = result_tuple[0];
    py::object seg_tensor = result_tuple[1];
    py::object conf_tensor = result_tuple[2];  // NEW: confidence (may be None)
    height = result_tuple[3].cast<int>();
    width = result_tuple[4].cast<int>();
    timestamp = result_tuple[5].cast<double>();

    py::object trav_numpy = trav_tensor.attr("numpy")();
    py::object seg_numpy = seg_tensor.attr("numpy")();

    py::buffer_info trav_buf = trav_numpy.cast<py::array>().request();
    py::buffer_info seg_buf = seg_numpy.cast<py::array>().request();

    if (trav_buf.ndim != 2 || seg_buf.ndim != 2) {
      std::fprintf(stderr, "[sem_map][bindings] read_semantic_results: expected 2D arrays\n");
      return false;
    }

    size_t total_elements = static_cast<size_t>(height * width);
    trav_out.resize(total_elements);
    seg_out.resize(total_elements);
    std::memcpy(trav_out.data(), trav_buf.ptr, total_elements * sizeof(float));
    std::memcpy(seg_out.data(), seg_buf.ptr, total_elements * sizeof(float));

    // Handle confidence (may be None for old models)
    if (!conf_tensor.is_none()) {
      py::object conf_numpy = conf_tensor.attr("numpy")();
      py::buffer_info conf_buf = conf_numpy.cast<py::array>().request();

      if (conf_buf.ndim == 2) {
        conf_out.resize(total_elements);
        std::memcpy(conf_out.data(), conf_buf.ptr, total_elements * sizeof(float));
      } else {
        conf_out.clear();  // Invalid confidence data
      }
    } else {
      conf_out.clear();  // No confidence available (old model)
    }

    static int success_count = 0;
    if (++success_count % 100 == 0) {
      std::fprintf(
          stderr,
          "[sem_map][GIL] read_semantic_results: wait=%ldμs hold=%ldμs total=%ldμs (n=%d)\n",
          gil_wait_us, gil_hold_us, total_us, success_count);
    }

    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[sem_map][bindings] read_semantic_results_from_buffer error: %s\n",
                 e.what());
    return false;
  }
}

bool SemMapWrapper::is_multiprocessing_enabled() const {
  return py_objects_->use_multiprocessing_;
}

}  // namespace sem_map
