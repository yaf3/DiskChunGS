#include "include/render_flythrough.h"

/**
 * Apply a smoothing filter to the path to reduce jerkiness
 * Uses a weighted moving average to smooth the path
 */
void smoothPath(std::vector<Eigen::Vector3d>& path, float smoothness_factor) {
  if (path.size() < 3) return;

  // Higher smoothness_factor = more aggressive smoothing
  // Clamp to [0,1] range
  smoothness_factor = std::max(0.0f, std::min(1.0f, smoothness_factor));

  // Calculate window size based on smoothness (larger window = smoother path)
  int window_size = 2 + static_cast<int>(smoothness_factor * 10);
  window_size = std::min(window_size, static_cast<int>(path.size() / 2));

  // Create a temporary copy of the path
  std::vector<Eigen::Vector3d> smoothed_path = path;

  // Apply multiple smoothing passes for stronger effect with higher
  // smoothness_factor
  int num_passes = 1 + static_cast<int>(smoothness_factor * 5);

  std::cout << "Applying path smoothing with " << num_passes
            << " passes and window size " << window_size << std::endl;

  for (int pass = 0; pass < num_passes; pass++) {
    for (size_t i = 0; i < path.size(); i++) {
      // Skip endpoints to preserve overall path shape
      if (i == 0 || i == path.size() - 1) continue;

      Eigen::Vector3d sum = Eigen::Vector3d::Zero();
      double total_weight = 0.0;

      // Calculate weighted average of neighboring points
      for (int j = -window_size; j <= window_size; j++) {
        int idx = static_cast<int>(i) + j;
        if (idx >= 0 && idx < static_cast<int>(path.size())) {
          // Gaussian-like weighting
          double weight = std::exp(-0.5 * j * j / (window_size / 2.0));
          sum += weight * path[idx];
          total_weight += weight;
        }
      }

      if (total_weight > 0) {
        smoothed_path[i] = sum / total_weight;
      }
    }

    // Update path for next pass
    path = smoothed_path;
  }
}

/**
 * Creates a smooth path through the given points using Catmull-Rom splines
 * @param keypoints The keyframe positions
 * @param points_per_segment Number of points to generate per segment
 * @param tension Controls how tightly the spline follows control points
 * (0.0-1.0) Lower values = smoother path (0.5 = standard Catmull-Rom)
 */
std::vector<Eigen::Vector3d> createSmoothPath(
    const std::vector<Eigen::Vector3d>& keypoints,
    int points_per_segment,
    float tension) {
  std::vector<Eigen::Vector3d> path;
  if (keypoints.size() < 2) return keypoints;

  // For only 2 points, do linear interpolation
  if (keypoints.size() == 2) {
    for (int i = 0; i <= points_per_segment; i++) {
      double t = static_cast<double>(i) / points_per_segment;
      path.push_back(keypoints[0] * (1 - t) + keypoints[1] * t);
    }
    return path;
  }

  // Create extended points array with extrapolated endpoints
  // This handles boundary conditions for Catmull-Rom
  std::vector<Eigen::Vector3d> extended;
  extended.push_back(keypoints[0] * 2 - keypoints[1]);  // Extrapolate start
  extended.insert(extended.end(), keypoints.begin(), keypoints.end());
  extended.push_back(keypoints.back() * 2 -
                     keypoints[keypoints.size() - 2]);  // Extrapolate end

  // Interpolate each segment
  for (size_t i = 1; i < extended.size() - 2; i++) {
    for (int j = 0; j < points_per_segment; j++) {
      double t = static_cast<double>(j) / points_per_segment;
      path.push_back(catmullRomInterpolate(extended[i - 1], extended[i],
                                           extended[i + 1], extended[i + 2], t,
                                           tension));
    }
  }

  // Add the final point
  path.push_back(keypoints.back());

  return path;
}

/**
 * Catmull-Rom spline interpolation for a single point
 */
Eigen::Vector3d catmullRomInterpolate(const Eigen::Vector3d& p0,
                                      const Eigen::Vector3d& p1,
                                      const Eigen::Vector3d& p2,
                                      const Eigen::Vector3d& p3,
                                      double t,
                                      float tension) {
  // Adjust tension to make curve looser (lower values) or tighter (higher
  // values) Standard Catmull-Rom uses tension = 0.5
  float alpha =
      1.0f - tension;  // Invert so higher smoothness_factor = looser curve

  double t2 = t * t;
  double t3 = t2 * t;

  // Catmull-Rom basis functions with tension control
  double h1 = -alpha * t3 + 2 * alpha * t2 - alpha * t;
  double h2 = (2 - alpha) * t3 + (alpha - 3) * t2 + 1.0;
  double h3 = (alpha - 2) * t3 + (3 - 2 * alpha) * t2 + alpha * t;
  double h4 = alpha * t3 - alpha * t2;

  return h1 * p0 + h2 * p1 + h3 * p2 + h4 * p3;
}

/**
 * Compute arc lengths along a path
 */
std::vector<double> computeArcLengths(
    const std::vector<Eigen::Vector3d>& path) {
  std::vector<double> arc_lengths(path.size(), 0.0);
  for (size_t i = 1; i < path.size(); i++) {
    double segment_length = (path[i] - path[i - 1]).norm();
    arc_lengths[i] = arc_lengths[i - 1] + segment_length;
  }
  return arc_lengths;
}
/**
 * Sample the path at equal distances and interpolate orientations
 */
void samplePathConstantSpeed(
    const std::vector<Eigen::Vector3d>& path,
    const std::vector<Eigen::Vector3d>& keyframe_positions,
    const std::vector<Eigen::Quaterniond>& keyframe_orientations,
    int num_samples,
    std::vector<Eigen::Vector3d>& sampled_positions,
    std::vector<Eigen::Quaterniond>& sampled_orientations) {
  sampled_positions.clear();
  sampled_orientations.clear();

  if (path.empty() || keyframe_positions.empty() ||
      keyframe_orientations.empty()) {
    return;
  }

  // 1. Compute arc lengths
  std::vector<double> arc_lengths = computeArcLengths(path);
  double total_length = arc_lengths.back();

  // 2. Map keyframes to path parameters
  std::vector<double> keyframe_parameters;
  mapKeyframesToPath(keyframe_positions, path, arc_lengths,
                     keyframe_parameters);

  // 3. Sample at equal distances
  for (int i = 0; i < num_samples; i++) {
    double t = static_cast<double>(i) / (num_samples - 1);  // Normalized [0,1]
    double target_length = total_length * t;

    // Position at this arc length
    Eigen::Vector3d position =
        samplePositionAtArcLength(path, arc_lengths, target_length);

    // Path parameter
    double path_param = target_length / total_length;

    // Interpolate orientation
    Eigen::Quaterniond orientation = interpolateOrientation(
        path_param, keyframe_parameters, keyframe_orientations);

    sampled_positions.push_back(position);
    sampled_orientations.push_back(orientation);
  }
}

/**
 * Map keyframe positions to their closest corresponding points on the path
 */
void mapKeyframesToPath(const std::vector<Eigen::Vector3d>& keyframe_positions,
                        const std::vector<Eigen::Vector3d>& path,
                        const std::vector<double>& arc_lengths,
                        std::vector<double>& keyframe_parameters) {
  keyframe_parameters.clear();
  double total_length = arc_lengths.back();

  for (const auto& kf_pos : keyframe_positions) {
    // Find closest point on path
    size_t closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (size_t i = 0; i < path.size(); i++) {
      double dist = (kf_pos - path[i]).squaredNorm();
      if (dist < min_dist) {
        min_dist = dist;
        closest_idx = i;
      }
    }

    // Parameter is normalized arc length
    double param = arc_lengths[closest_idx] / total_length;
    keyframe_parameters.push_back(param);
  }

  // Ensure parameters are strictly increasing (required for interpolation)
  for (size_t i = 1; i < keyframe_parameters.size(); i++) {
    if (keyframe_parameters[i] <= keyframe_parameters[i - 1]) {
      keyframe_parameters[i] = keyframe_parameters[i - 1] + 0.001;
    }
  }
}

/**
 * Sample a position at a specific arc length along the path
 */
Eigen::Vector3d samplePositionAtArcLength(
    const std::vector<Eigen::Vector3d>& path,
    const std::vector<double>& arc_lengths,
    double target_length) {
  // Find segment containing this arc length
  auto it =
      std::lower_bound(arc_lengths.begin(), arc_lengths.end(), target_length);
  int idx = std::distance(arc_lengths.begin(), it);

  if (idx >= path.size()) {
    return path.back();  // Beyond the end
  } else if (idx == 0) {
    return path.front();  // Before the start
  } else {
    // Interpolate within segment
    double segment_start = arc_lengths[idx - 1];
    double segment_length = arc_lengths[idx] - segment_start;
    double t = segment_length > 0
                   ? (target_length - segment_start) / segment_length
                   : 0;

    return path[idx - 1] * (1 - t) + path[idx] * t;
  }
}

/**
 * Interpolate orientation using SLERP based on path parameter
 */
Eigen::Quaterniond interpolateOrientation(
    double param,
    const std::vector<double>& keyframe_parameters,
    const std::vector<Eigen::Quaterniond>& keyframe_orientations) {
  // Handle boundary cases
  if (param <= keyframe_parameters.front()) {
    return keyframe_orientations.front();
  }
  if (param >= keyframe_parameters.back()) {
    return keyframe_orientations.back();
  }

  // Find the keyframes before and after this parameter
  size_t idx = 0;
  while (idx < keyframe_parameters.size() - 1 &&
         keyframe_parameters[idx + 1] < param) {
    idx++;
  }

  // SLERP between these orientations
  double segment_length =
      keyframe_parameters[idx + 1] - keyframe_parameters[idx];
  double t = segment_length > 0
                 ? (param - keyframe_parameters[idx]) / segment_length
                 : 0;
  t = std::max(0.0, std::min(1.0, t));  // Clamp to [0,1]

  return keyframe_orientations[idx].slerp(t, keyframe_orientations[idx + 1]);
}

/**
 * Calculate a rough estimate of scene scale based on keyframe positions
 */
double calculateSceneScale(const std::vector<Eigen::Vector3d>& positions) {
  if (positions.size() < 2) {
    return 1.0;  // Default scale if not enough positions
  }

  // Compute bounding box
  Eigen::Vector3d min_point = positions[0];
  Eigen::Vector3d max_point = positions[0];

  for (const auto& pos : positions) {
    min_point = min_point.cwiseMin(pos);
    max_point = max_point.cwiseMax(pos);
  }

  // Use diagonal length as scene scale
  double diag_length = (max_point - min_point).norm();
  return diag_length / 20.0;  // Scale to reasonable deviation size
}

/**
 * Compute orthonormal frames along the path for consistent orientation
 */
std::vector<Eigen::Matrix3d> computePathFrames(
    const std::vector<Eigen::Vector3d>& path) {
  std::vector<Eigen::Matrix3d> frames(path.size());

  if (path.size() < 2) {
    // Default frame if path is too short
    for (size_t i = 0; i < path.size(); i++) {
      frames[i] = Eigen::Matrix3d::Identity();
    }
    return frames;
  }

  // First pass: compute tangent directions (forward vectors)
  std::vector<Eigen::Vector3d> tangents(path.size());

  // First point (forward difference)
  tangents[0] = (path[1] - path[0]).normalized();

  // Middle points (central difference)
  for (size_t i = 1; i < path.size() - 1; i++) {
    tangents[i] = (path[i + 1] - path[i - 1]).normalized();
  }

  // Last point (backward difference)
  tangents[path.size() - 1] =
      (path[path.size() - 1] - path[path.size() - 2]).normalized();

  // Second pass: build frames using parallel transport
  // Initial frame: use world up as reference if possible
  Eigen::Vector3d world_up(0, 1, 0);
  Eigen::Vector3d t0 = tangents[0];
  Eigen::Vector3d right0 = t0.cross(world_up);

  // If right vector is too small, world_up is nearly parallel to tangent
  // In this case, use a different reference
  if (right0.norm() < 1e-6) {
    world_up = Eigen::Vector3d(0, 0, 1);
    right0 = t0.cross(world_up);

    // If still problematic, try x-axis
    if (right0.norm() < 1e-6) {
      world_up = Eigen::Vector3d(1, 0, 0);
      right0 = t0.cross(world_up);
    }
  }

  right0.normalize();
  Eigen::Vector3d up0 = right0.cross(t0).normalized();

  // Set first frame
  frames[0].col(0) = right0;  // right
  frames[0].col(1) = up0;     // up
  frames[0].col(2) = t0;      // forward

  // Propagate the frame along the curve
  for (size_t i = 1; i < path.size(); i++) {
    Eigen::Vector3d prev_t = tangents[i - 1];
    Eigen::Vector3d curr_t = tangents[i];

    // Compute rotation from prev_t to curr_t
    Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(prev_t, curr_t);

    // Rotate previous frame
    Eigen::Vector3d prev_right = frames[i - 1].col(0);
    Eigen::Vector3d prev_up = frames[i - 1].col(1);

    Eigen::Vector3d curr_right = q * prev_right;
    Eigen::Vector3d curr_up = q * prev_up;

    // Ensure frame is orthogonal
    curr_right = curr_right - curr_right.dot(curr_t) * curr_t;
    curr_right.normalize();
    curr_up = curr_t.cross(curr_right).normalized();

    // Set frame
    frames[i].col(0) = curr_right;  // right
    frames[i].col(1) = curr_up;     // up
    frames[i].col(2) = curr_t;      // forward
  }

  return frames;
}