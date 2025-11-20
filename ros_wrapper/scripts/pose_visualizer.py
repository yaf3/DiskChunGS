#!/usr/bin/env python

import rospy
import tf2_ros
import geometry_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np

class PoseVisualizer:
    def __init__(self):
        rospy.init_node('pose_visualizer')
        
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(30.0))  # Increased buffer time
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publisher for trajectory and camera frustum visualization
        self.marker_pub = rospy.Publisher('/camera_visualization', MarkerArray, queue_size=10)
        self.trajectory = []
        
        # Parameters
        self.target_frame = rospy.get_param('~target_frame', 'map')
        self.source_frame = rospy.get_param('~source_frame', 'zed2i_left_camera_optical_frame')
        
        # Camera frustum parameters
        self.frustum_scale = 0.2

        # Wait for TF tree to become available
        rospy.loginfo(f"Waiting for transform between {self.target_frame} and {self.source_frame}...")
        while not rospy.is_shutdown():
            try:
                self.tf_buffer.lookup_transform(self.target_frame, self.source_frame, rospy.Time(), rospy.Duration(1.0))
                rospy.loginfo("Transform found!")
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(1, f"Waiting for transform... ({e})")
                rospy.sleep(0.1)
                continue

    def create_camera_frustum_marker(self, trans):
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "camera_frustum"
        marker.id = 2
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.01  # line width
        marker.color.b = 1.0  # Blue color for frustum
        marker.color.a = 1.0

        # Camera position
        pos = np.array([trans.transform.translation.x,
                       trans.transform.translation.y,
                       trans.transform.translation.z])

        # Convert quaternion to rotation matrix
        q = trans.transform.rotation
        rot_matrix = np.array([
            [1 - 2*q.y*q.y - 2*q.z*q.z, 2*q.x*q.y - 2*q.z*q.w, 2*q.x*q.z + 2*q.y*q.w],
            [2*q.x*q.y + 2*q.z*q.w, 1 - 2*q.x*q.x - 2*q.z*q.z, 2*q.y*q.z - 2*q.x*q.w],
            [2*q.x*q.z - 2*q.y*q.w, 2*q.y*q.z + 2*q.x*q.w, 1 - 2*q.x*q.x - 2*q.y*q.y]
        ])

        # Define frustum corners in camera frame
        s = self.frustum_scale
        corners = np.array([
            [0, 0, 0],  # Camera center
            [s, s, s],  # Top right
            [s, -s, s],  # Bottom right
            [-s, -s, s],  # Bottom left
            [-s, s, s],  # Top left
        ])

        # Transform corners to world frame
        corners_world = [pos + rot_matrix @ corner for corner in corners]

        # Create lines for frustum
        center = corners_world[0]
        for i in range(1, 5):
            # Line from center to corner
            marker.points.append(Point(*center))
            marker.points.append(Point(*corners_world[i]))
            
            # Lines between corners
            marker.points.append(Point(*corners_world[i]))
            marker.points.append(Point(*corners_world[1 if i == 4 else i + 1]))

        return marker

    def create_visualization_markers(self, trans):
        marker_array = MarkerArray()
        
        # Trajectory line strip
        line_strip = Marker()
        line_strip.header.frame_id = self.target_frame
        line_strip.header.stamp = rospy.Time.now()
        line_strip.ns = "camera_trajectory"
        line_strip.id = 0
        line_strip.type = Marker.LINE_STRIP
        line_strip.action = Marker.ADD
        line_strip.scale.x = 0.02
        line_strip.color.r = 1.0
        line_strip.color.a = 1.0
        line_strip.points = [Point(p[0], p[1], p[2]) for p in self.trajectory]
        
        marker_array.markers.append(line_strip)
        
        # Current position and frustum
        if self.trajectory:
            # Add camera frustum
            frustum_marker = self.create_camera_frustum_marker(trans)
            marker_array.markers.append(frustum_marker)
            
            # Current position sphere
            current_pos = Marker()
            current_pos.header.frame_id = self.target_frame
            current_pos.header.stamp = rospy.Time.now()
            current_pos.ns = "current_position"
            current_pos.id = 1
            current_pos.type = Marker.SPHERE
            current_pos.action = Marker.ADD
            current_pos.scale.x = current_pos.scale.y = current_pos.scale.z = 0.05
            current_pos.color.g = 1.0
            current_pos.color.a = 1.0
            current_pos.pose.position.x = self.trajectory[-1][0]
            current_pos.pose.position.y = self.trajectory[-1][1]
            current_pos.pose.position.z = self.trajectory[-1][2]
            
            marker_array.markers.append(current_pos)
        
        return marker_array

    def run(self):
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    self.source_frame,
                    rospy.Time(0)
                )
                
                # Extract position
                pos = [
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    trans.transform.translation.z
                ]
                
                # Add to trajectory
                self.trajectory.append(pos)
                
                # Create and publish markers
                marker_array = self.create_visualization_markers(trans)
                self.marker_pub.publish(marker_array)
                
                # Print pose information for debugging
                q = trans.transform.rotation
                rospy.loginfo_throttle(1, f"Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
                rospy.loginfo_throttle(1, f"Orientation (quat): x={q.x:.3f}, y={q.y:.3f}, z={q.z:.3f}, w={q.w:.3f}")
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn_throttle(1, f"TF Error: {e}")
                
            rate.sleep()

if __name__ == '__main__':
    try:
        visualizer = PoseVisualizer()
        visualizer.run()
    except rospy.ROSInterruptException:
        pass