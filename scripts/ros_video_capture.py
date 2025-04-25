#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import os
import time
import subprocess
import tempfile
import shutil

class FrameCapture:
    def __init__(self, topic_name, num_frames, output_file, fps=30):
        """
        Initialize the frame capture class.
        
        Args:
            topic_name (str): Name of the ROS topic to subscribe to
            num_frames (int): Number of frames to capture
            output_file (str): Path to save the output MP4 file
            fps (int): Frames per second for the output video
        """
        self.bridge = CvBridge()
        self.frames_to_capture = num_frames
        self.captured_frames = []
        self.output_file = output_file
        self.fps = fps
        self.is_done = False
        self.temp_dir = tempfile.mkdtemp()
        
        rospy.loginfo(f"Created temporary directory for frames: {self.temp_dir}")
        
        # Initialize the ROS node
        rospy.init_node('frame_capture', anonymous=True)
        
        # Create a subscriber to the image topic
        self.image_sub = rospy.Subscriber(topic_name, Image, self.callback)
        
        rospy.loginfo(f"Starting to capture {num_frames} frames from {topic_name}")
        
    def callback(self, data):
        """
        Callback function for the image subscriber.
        
        Args:
            data (Image): ROS Image message
        """
        if len(self.captured_frames) < self.frames_to_capture and not self.is_done:
            try:
                # Convert ROS Image message to OpenCV image
                # Explicitly specify the expected encoding based on the actual message
                encoding = data.encoding if data.encoding else "bgr8"
                cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding=encoding)
                
                frame_number = len(self.captured_frames)
                frame_path = os.path.join(self.temp_dir, f"frame_{frame_number:06d}.png")
                
                # Save the frame as a PNG file
                cv2.imwrite(frame_path, cv_image)
                
                # Store the frame path
                self.captured_frames.append(frame_path)
                
                # Log progress
                rospy.loginfo(f"Captured frame {frame_number+1}/{self.frames_to_capture} - Size: {cv_image.shape[1]}x{cv_image.shape[0]}")
                
                # Check if we've captured all frames
                if len(self.captured_frames) >= self.frames_to_capture:
                    self.is_done = True
                    self.create_video_with_ffmpeg()
                    rospy.signal_shutdown("Finished capturing frames")
                    
            except Exception as e:
                rospy.logerr(f"Error processing frame: {e}")
    
    def create_video_with_ffmpeg(self):
        """
        Use FFmpeg to create a video from the captured frames.
        """
        if not self.captured_frames:
            rospy.logerr("No frames captured. Cannot create video.")
            return
        
        try:
            # Create a text file listing all frames for FFmpeg
            frames_list_file = os.path.join(self.temp_dir, "frames_list.txt")
            with open(frames_list_file, 'w') as f:
                for frame_path in self.captured_frames:
                    f.write(f"file '{frame_path}'\n")
            
            # FFmpeg command to create video
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-r', str(self.fps),  # Frame rate
                '-f', 'concat',  # Input format
                '-safe', '0',  # Allow absolute paths
                '-i', frames_list_file,  # Input file list
                '-c:v', 'libx264',  # Codec
                '-pix_fmt', 'yuv420p',  # Pixel format
                '-preset', 'medium',  # Encoding speed preset
                '-crf', '23',  # Quality (lower is better)
                self.output_file  # Output file
            ]
            
            rospy.loginfo(f"Creating video with FFmpeg: {' '.join(ffmpeg_cmd)}")
            
            # Run FFmpeg
            process = subprocess.Popen(
                ffmpeg_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                rospy.logerr(f"FFmpeg error: {stderr.decode()}")
                # Try alternative approach
                self.create_video_with_ffmpeg_alt()
            else:
                rospy.loginfo(f"Video created successfully: {self.output_file}")
                
        except Exception as e:
            rospy.logerr(f"Error creating video with FFmpeg: {e}")
            # Try alternative approach
            self.create_video_with_ffmpeg_alt()
        finally:
            # Clean up temporary files
            self.cleanup()
    
    def create_video_with_ffmpeg_alt(self):
        """
        Alternative approach using FFmpeg's image2 demuxer
        """
        try:
            # Use a pattern for the frame filenames
            frame_pattern = os.path.join(self.temp_dir, "frame_%06d.png")
            
            # FFmpeg command using image2 demuxer
            ffmpeg_cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file if it exists
                '-framerate', str(self.fps),  # Input frame rate
                '-i', frame_pattern,  # Input pattern
                '-c:v', 'libx264',  # Codec
                '-pix_fmt', 'yuv420p',  # Pixel format
                '-preset', 'medium',  # Encoding speed preset
                '-crf', '23',  # Quality (lower is better)
                self.output_file  # Output file
            ]
            
            rospy.loginfo(f"Trying alternative FFmpeg approach: {' '.join(ffmpeg_cmd)}")
            
            process = subprocess.Popen(
                ffmpeg_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                rospy.logerr(f"Alternative FFmpeg approach failed: {stderr.decode()}")
            else:
                rospy.loginfo(f"Video created successfully with alternative approach: {self.output_file}")
                
        except Exception as e:
            rospy.logerr(f"Error in alternative FFmpeg approach: {e}")
    
    def cleanup(self):
        """
        Clean up temporary files
        """
        try:
            rospy.loginfo(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            rospy.logwarn(f"Error cleaning up temporary files: {e}")

def check_ffmpeg():
    """
    Check if FFmpeg is installed.
    """
    try:
        subprocess.run(['ffmpeg', '-version'], 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Capture frames from a ROS topic and save as MP4 using FFmpeg')
    parser.add_argument('--topic', type=str, default='/boxi/zed2i/left/image_raw_uncompressed',
                        help='ROS topic to subscribe to')
    parser.add_argument('--frames', type=int, default=100,
                        help='Number of frames to capture')
    parser.add_argument('--output', type=str, default='output.mp4',
                        help='Output MP4 file path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for output video')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        rospy.loginfo("Debug mode enabled - setting log level to DEBUG")
        rospy.set_param('/rosout/log_level', 'DEBUG')
    
    # Check for FFmpeg
    if not check_ffmpeg():
        rospy.logerr("FFmpeg is not installed or not in PATH. Please install FFmpeg.")
        return
    
    # Create the frame capture object
    frame_capture = FrameCapture(
        topic_name=args.topic,
        num_frames=args.frames,
        output_file=args.output,
        fps=args.fps
    )
    
    try:
        # Keep the script running until we've captured all frames
        while not frame_capture.is_done and not rospy.is_shutdown():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down due to keyboard interrupt")
        
        # Create video with any frames we've captured so far
        if frame_capture.captured_frames:
            frame_capture.create_video_with_ffmpeg()

if __name__ == '__main__':
    main()