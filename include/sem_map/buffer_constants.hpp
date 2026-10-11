// Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
#ifndef SEM_MAP__BUFFER_CONSTANTS_HPP_
#define SEM_MAP__BUFFER_CONSTANTS_HPP_

namespace sem_map {
namespace buffer_sizes {

// Image/traversability buffer sizes (based on typical camera resolutions)
constexpr size_t MAX_IMAGE_WIDTH = 1920;
constexpr size_t MAX_IMAGE_HEIGHT = 1080;
constexpr size_t MAX_IMAGE_PIXELS = MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT;
constexpr size_t MAX_IMAGE_BYTES = MAX_IMAGE_PIXELS * 3;  // RGB

// Traversability map (typically same as image resolution)
constexpr size_t MAX_TRAVERSABILITY_PIXELS = MAX_IMAGE_PIXELS;

// Pointcloud buffer sizes
constexpr size_t MAX_POINTCLOUD_POINTS = 100000;  // 100K points typical for lidar
constexpr size_t MAX_POINTCLOUD_XYZ = MAX_POINTCLOUD_POINTS * 3;

// Elevation map sizes (depends on map_size and resolution)
// With map_size=20m and resolution=0.05m: 400x400 = 160K cells
constexpr size_t MAX_ELEVATION_MAP_CELLS = 400 * 400;

}  // namespace buffer_sizes
}  // namespace sem_map

#endif  // SEM_MAP__BUFFER_CONSTANTS_HPP_
