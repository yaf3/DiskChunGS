#!/usr/bin/env python

import rospy
import os
import argparse
import numpy as np
import cv2
from cv_bridge import CvBridge
import tf2_ros
import subprocess
import time
import threading
import signal
from sensor_msgs.msg import CompressedImage
import geometry_msgs.msg

def extract_data_from_ros(output_dir, image_topic, target_frame, source_frame, done_event):
    """
    Extract images and ground truth poses from ROS topics
    
    Args:
        output_dir: Output directory for extracted data
        image_topic: Topic name for compressed images
        target_frame: Target frame for TF lookup (e.g., "map")
        source_frame: Source frame for TF lookup (e.g., "zed2_left_camera_optical_frame")
        done_event: Event to signal when extraction is complete
    """
    # Create output directories
    rgb_dir = os.path.join(output_dir, "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    
    # Set up TF buffer and listener
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3600))  # Large cache to hold all transforms
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    
    # Set up CV bridge for image conversion
    bridge = CvBridge()
    
    # Data storage
    images_received = 0
    poses_found = 0
    pose_data = []
    last_image_time = time.time()
    
    def image_callback(msg):
        nonlocal images_received, poses_found, last_image_time
        timestamp = msg.header.stamp.to_sec()
        last_image_time = time.time()
        
        # Convert compressed image
        try:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg)
            img_filename = f"{timestamp:.9f}.png"
            cv2.imwrite(os.path.join(rgb_dir, img_filename), cv_img)
            images_received += 1
            
            # Look up transform
            try:
                transform = tf_buffer.lookup_transform(target_frame, source_frame, msg.header.stamp)
                
                # Extract translation and rotation
                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                tz = transform.transform.translation.z
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                
                # Add to pose data
                pose_data.append(f"{timestamp:.9f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")
                poses_found += 1
                
                if images_received % 10 == 0:
                    print(f"Processed {images_received} images, found {poses_found} poses")
                
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                print(f"TF lookup error at timestamp {timestamp}: {e}")
                
        except Exception as e:
            print(f"Error processing image at timestamp {timestamp}: {e}")
    
    # Subscribe to image topic
    rospy.Subscriber(image_topic, CompressedImage, image_callback)
    
    # Wait until we're done
    # We'll consider extraction complete if we don't receive images for 5 seconds
    while not rospy.is_shutdown() and not done_event.is_set():
        rospy.sleep(0.5)
        if time.time() - last_image_time > 5.0 and images_received > 0:
            print("No new images for 5 seconds, finishing extraction")
            break
    
    # Write pose file in TUM format
    with open(os.path.join(output_dir, "groundtruth.txt"), 'w') as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for pose_line in pose_data:
            f.write(f"{pose_line}\n")
    
    print(f"Extracted {images_received} images and {poses_found} poses")
    print(f"Data saved to {output_dir}")
    
    # Signal that we're done
    done_event.set()

def main():
    parser = argparse.ArgumentParser(description="Extract data from ROS bag for SLAM evaluation")
    parser.add_argument("bag_path", help="Path to the ROS bag file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--image_topic", default="/zed2/zed_node/rgb/image_rect_color/compressed", 
                        help="Topic for compressed RGB images")
    parser.add_argument("--target_frame", default="map", 
                        help="Target frame for TF lookup")
    parser.add_argument("--source_frame", default="zed2_left_camera_optical_frame", 
                        help="Source frame for TF lookup")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Playback rate factor")
    
    args = parser.parse_args()
    
    # Initialize ROS node
    rospy.init_node('bag_extractor', anonymous=True)
    
    # Event to signal when extraction is complete
    done_event = threading.Event()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("Ctrl+C received, stopping extraction")
        done_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start extraction process
    extract_thread = threading.Thread(target=extract_data_from_ros, 
                                    args=(args.output_dir, args.image_topic, 
                                          args.target_frame, args.source_frame,
                                          done_event))
    extract_thread.daemon = True
    extract_thread.start()
    
    # Start bag playback in a separate process
    # Using --pause and then sending SIGCONT ensures that static TFs are published first
    print("Starting bag playback with pause to ensure static TFs are published...")
    bag_play_cmd = f"rosbag play --pause {args.bag_path} --rate {args.rate}"
    bag_process = subprocess.Popen(bag_play_cmd, shell=True)
    
    # Wait for ROS to initialize and TF to be ready
    print("Waiting for TF to initialize...")
    rospy.sleep(2.0)
    
    # Send SIGCONT to start bag playback
    print("Resuming bag playback...")
    subprocess.run(f"pkill -CONT -f 'rosbag play'", shell=True)
    
    # Wait for the extraction to complete or bag playback to finish
    try:
        while not done_event.is_set():
            if bag_process.poll() is not None:
                print("Bag playback finished")
                # Give some extra time for processing the last messages
                time.sleep(5.0)
                done_event.set()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Extraction interrupted")
        done_event.set()
    
    # Wait for extraction thread to finish
    extract_thread.join()
    
    # Clean up
    try:
        subprocess.run("pkill -f 'rosbag play'", shell=True)
    except:
        pass
    
    print("Extraction complete")

if __name__ == "__main__":
    main()