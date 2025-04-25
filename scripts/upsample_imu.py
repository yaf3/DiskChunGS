#!/usr/bin/env python3

import rosbag
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import os
import argparse
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
from tqdm import tqdm

def imu_upsampler(input_bag_path, output_bag_path, target_frequency=100.0, imu_topic='/boxi/zed2i/imu/data'):
    """
    Upsample IMU data in a rosbag to a higher frequency using interpolation
    
    Args:
        input_bag_path: Path to input rosbag
        output_bag_path: Path to save the output rosbag
        target_frequency: Target IMU frequency in Hz
        imu_topic: Topic name of the IMU messages
    """
    print(f"Reading input bag: {input_bag_path}")
    print(f"Target frequency: {target_frequency} Hz")
    print(f"IMU topic: {imu_topic}")
    
    # Collect original IMU data
    data = []
    
    # Read the original bag
    with rosbag.Bag(input_bag_path, 'r') as bag:
        # First pass: collect all IMU data
        for topic, msg, t in tqdm(bag.read_messages(topics=[imu_topic]), desc="Reading IMU messages"):
            timestamp = msg.header.stamp.to_sec()
            
            # Store all data in a tuple
            data.append((
                timestamp,
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
                msg.header.frame_id,
                msg.orientation_covariance,
                msg.angular_velocity_covariance,
                msg.linear_acceleration_covariance
            ))
    
    # Check for timestamp issues
    timestamps = np.array([d[0] for d in data])
    diffs = np.diff(timestamps)
    
    # Count negative time differences (out of order) and zero differences (duplicates)
    neg_diffs = np.sum(diffs < 0)
    zero_diffs = np.sum(diffs == 0)
    
    print(f"Found {neg_diffs} out-of-order timestamps and {zero_diffs} duplicate timestamps")
    
    if neg_diffs > 0 or zero_diffs > 0:
        print("Fixing timestamp issues...")
        
        # Sort data by timestamp
        data.sort(key=lambda x: x[0])
        
        # Remove duplicates by keeping only the first occurrence of each timestamp
        unique_data = []
        seen_timestamps = set()
        
        for item in data:
            timestamp = item[0]
            # Use 6 decimal precision to handle floating point comparison
            rounded_ts = round(timestamp, 6)
            
            if rounded_ts not in seen_timestamps:
                seen_timestamps.add(rounded_ts)
                unique_data.append(item)
                
        print(f"Removed {len(data) - len(unique_data)} problematic entries")
        data = unique_data
        
        # Update timestamps after cleaning
        timestamps = np.array([d[0] for d in data])
    
    # Extract all data components
    linear_accel_x = np.array([d[1] for d in data])
    linear_accel_y = np.array([d[2] for d in data])
    linear_accel_z = np.array([d[3] for d in data])
    angular_vel_x = np.array([d[4] for d in data])
    angular_vel_y = np.array([d[5] for d in data])
    angular_vel_z = np.array([d[6] for d in data])
    
    # Extract quaternions and convert to proper format
    orientations = np.array([[d[7], d[8], d[9], d[10]] for d in data])
    
    # Get frame_id and covariances from first message (assuming they don't change)
    frame_id = data[0][11]
    orientation_cov = data[0][12]
    angular_velocity_cov = data[0][13]
    linear_acceleration_cov = data[0][14]
    
    # Calculate time range and original frequency
    time_range = timestamps[-1] - timestamps[0]
    original_frequency = len(timestamps) / time_range
    print(f"Original data: {len(timestamps)} points over {time_range:.2f} seconds")
    print(f"Original frequency: {original_frequency:.2f} Hz")
    
    # Create new timestamps at target frequency
    delta_t = 1.0 / target_frequency
    new_timestamps = np.arange(timestamps[0], timestamps[-1], delta_t)
    
    print(f"Creating {len(new_timestamps)} interpolated points at {target_frequency} Hz")
    
    # Interpolate linear acceleration
    accel_x_interp = interp1d(timestamps, linear_accel_x, kind='linear', fill_value='extrapolate')
    accel_y_interp = interp1d(timestamps, linear_accel_y, kind='linear', fill_value='extrapolate')
    accel_z_interp = interp1d(timestamps, linear_accel_z, kind='linear', fill_value='extrapolate')
    
    # Interpolate angular velocity
    ang_vel_x_interp = interp1d(timestamps, angular_vel_x, kind='linear', fill_value='extrapolate')
    ang_vel_y_interp = interp1d(timestamps, angular_vel_y, kind='linear', fill_value='extrapolate')
    ang_vel_z_interp = interp1d(timestamps, angular_vel_z, kind='linear', fill_value='extrapolate')
    
    # Interpolate orientation using Slerp
    # Create rotation objects
    rotations = Rotation.from_quat(orientations)
    
    # Create Slerp object
    slerp = Slerp(timestamps, rotations)
    
    # Get interpolated values
    new_accel_x = accel_x_interp(new_timestamps)
    new_accel_y = accel_y_interp(new_timestamps)
    new_accel_z = accel_z_interp(new_timestamps)
    
    new_ang_vel_x = ang_vel_x_interp(new_timestamps)
    new_ang_vel_y = ang_vel_y_interp(new_timestamps)
    new_ang_vel_z = ang_vel_z_interp(new_timestamps)
    
    new_orientations = slerp(new_timestamps).as_quat()
    
    # Create output bag and write all messages (including non-IMU)
    with rosbag.Bag(input_bag_path, 'r') as inbag, \
         rosbag.Bag(output_bag_path, 'w') as outbag:
        
        # Copy all non-IMU messages
        for topic, msg, t in tqdm(inbag.read_messages(topics=[x for x in inbag.get_type_and_topic_info().topics.keys() if x != imu_topic]), 
                                 desc="Copying non-IMU messages"):
            outbag.write(topic, msg, t)
        
        # Write new IMU messages
        for i, timestamp in enumerate(tqdm(new_timestamps, desc="Writing upsampled IMU data")):
            # Create new IMU message
            imu_msg = Imu()
            
            # Set header
            imu_msg.header.stamp.secs = int(timestamp)
            imu_msg.header.stamp.nsecs = int((timestamp - int(timestamp)) * 1e9)
            imu_msg.header.frame_id = frame_id
            
            # Set linear acceleration
            imu_msg.linear_acceleration.x = new_accel_x[i]
            imu_msg.linear_acceleration.y = new_accel_y[i]
            imu_msg.linear_acceleration.z = new_accel_z[i]
            
            # Set angular velocity
            imu_msg.angular_velocity.x = new_ang_vel_x[i]
            imu_msg.angular_velocity.y = new_ang_vel_y[i]
            imu_msg.angular_velocity.z = new_ang_vel_z[i]
            
            # Set orientation
            imu_msg.orientation.x = new_orientations[i][0]
            imu_msg.orientation.y = new_orientations[i][1]
            imu_msg.orientation.z = new_orientations[i][2]
            imu_msg.orientation.w = new_orientations[i][3]
            
            # Set covariances (using original values)
            imu_msg.orientation_covariance = orientation_cov
            imu_msg.angular_velocity_covariance = angular_velocity_cov
            imu_msg.linear_acceleration_covariance = linear_acceleration_cov
            
            # Get ROS time object for bag write
            ros_time = rospy.Time(secs=int(timestamp), nsecs=int((timestamp - int(timestamp)) * 1e9))
            
            # Write message to bag
            outbag.write(imu_topic, imu_msg, ros_time)
    
    print(f"Upsampled IMU data written to {output_bag_path}")
    print(f"Number of IMU messages increased from {len(timestamps)} to {len(new_timestamps)}")

if __name__ == '__main__':
    import rospy
    
    parser = argparse.ArgumentParser(description='Upsample IMU data in a ROS bag file')
    parser.add_argument('input_bag', help='Input ROS bag file')
    parser.add_argument('output_bag', help='Output ROS bag file')
    parser.add_argument('--freq', type=float, default=100.0, help='Target frequency in Hz (default: 100.0)')
    parser.add_argument('--topic', default='/boxi/zed2i/imu/data', help='IMU topic (default: /boxi/zed2i/imu/data)')
    
    args = parser.parse_args()
    
    imu_upsampler(args.input_bag, args.output_bag, args.freq, args.topic)