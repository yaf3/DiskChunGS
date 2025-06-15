import cv2
import os
import argparse
from pathlib import Path

def extract_frames_from_video(video_path, output_dir, frame_interval=1, start_frame=0, end_frame=None, image_format='png'):
    """
    Extract frames from a video file and save them as numbered images.
    
    Args:
        video_path (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract every Nth frame (1 = every frame, 2 = every other frame, etc.)
        start_frame (int): Frame number to start extraction from
        end_frame (int): Frame number to stop extraction at (None = until end)
        image_format (str): Output image format ('png', 'jpg', 'bmp', etc.)
    """
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Set start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Set end frame
    if end_frame is None:
        end_frame = total_frames
    else:
        end_frame = min(end_frame, total_frames)
    
    print(f"\nExtracting frames {start_frame} to {end_frame-1} (every {frame_interval} frame(s))")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {image_format}")
    
    frame_count = 0
    saved_count = 0
    current_frame = start_frame
    
    while True:
        ret, frame = cap.read()
        
        if not ret or current_frame >= end_frame:
            break
        
        # Save frame if it matches our interval
        if (current_frame - start_frame) % frame_interval == 0:
            # Generate filename with zero-padded numbering
            filename = f"{saved_count:06d}.{image_format}"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            success = cv2.imwrite(filepath, frame)
            
            if success:
                saved_count += 1
                if saved_count % 100 == 0:  # Progress update every 100 frames
                    print(f"  Saved {saved_count} frames...")
            else:
                print(f"  Error saving frame {current_frame} to {filepath}")
        
        current_frame += 1
        frame_count += 1
    
    cap.release()
    
    print(f"\nCompleted!")
    print(f"  Processed {frame_count} frames")
    print(f"  Saved {saved_count} images")
    print(f"  Images saved to: {output_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video file')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('-o', '--output', default='frames', help='Output directory (default: frames)')
    parser.add_argument('-i', '--interval', type=int, default=1, help='Frame interval (extract every Nth frame, default: 1)')
    parser.add_argument('-s', '--start', type=int, default=0, help='Start frame number (default: 0)')
    parser.add_argument('-e', '--end', type=int, default=None, help='End frame number (default: end of video)')
    parser.add_argument('-f', '--format', default='png', choices=['png', 'jpg', 'jpeg', 'bmp', 'tiff'], 
                       help='Output image format (default: png)')
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} does not exist!")
        return
    
    # Extract frames
    success = extract_frames_from_video(
        video_path=args.video_path,
        output_dir=args.output,
        frame_interval=args.interval,
        start_frame=args.start,
        end_frame=args.end,
        image_format=args.format
    )
    
    if success:
        print(f"\n✓ Frame extraction completed successfully!")
    else:
        print(f"\n✗ Frame extraction failed!")

# ====================================================================
# EXAMPLE USAGE FUNCTIONS
# ====================================================================

def extract_all_frames(video_path, output_dir="frames"):
    """Simple function to extract all frames from a video"""
    return extract_frames_from_video(video_path, output_dir)

def extract_every_nth_frame(video_path, n=10, output_dir="frames"):
    """Extract every Nth frame (useful for large videos)"""
    return extract_frames_from_video(video_path, output_dir, frame_interval=n)

def extract_frames_time_range(video_path, start_seconds=0, end_seconds=None, output_dir="frames"):
    """Extract frames from a specific time range"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps) if end_seconds else None
    
    return extract_frames_from_video(video_path, output_dir, start_frame=start_frame, end_frame=end_frame)

def extract_frames_for_calibration(video_path, output_dir="calibration_frames", interval=30):
    """
    Extract frames specifically for camera calibration.
    Uses larger interval to get diverse frames.
    """
    print("Extracting frames for camera calibration...")
    print("Tip: Make sure your checkerboard is visible and clear in the video!")
    
    return extract_frames_from_video(video_path, output_dir, frame_interval=interval)

# ====================================================================
# INTERACTIVE MODE
# ====================================================================

def interactive_mode():
    """Interactive mode for easy usage"""
    print("=== Video to Frames Extractor ===")
    print()
    
    # Get video path
    video_path = input("Enter video file path: ").strip('"')
    
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist!")
        return
    
    # Get output directory
    output_dir = input("Enter output directory (default: frames): ").strip() or "frames"
    
    # Get frame interval
    try:
        interval = int(input("Extract every Nth frame (1=all frames, 10=every 10th, etc., default: 1): ") or "1")
    except ValueError:
        interval = 1
    
    # Get image format
    format_choice = input("Image format (png/jpg, default: png): ").strip().lower() or "png"
    if format_choice not in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        format_choice = 'png'
    
    print(f"\nStarting extraction...")
    print(f"Video: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Interval: every {interval} frame(s)")
    print(f"Format: {format_choice}")
    print()
    
    success = extract_frames_from_video(
        video_path=video_path,
        output_dir=output_dir,
        frame_interval=interval,
        image_format=format_choice
    )
    
    if success:
        print(f"\n✓ Extraction completed!")
        print(f"Frames saved in: {output_dir}")
    else:
        print(f"\n✗ Extraction failed!")

# ====================================================================
# EXAMPLE USAGE
# ====================================================================

if __name__ == "__main__":
    import sys
    
    # If run with command line arguments, use argument parser
    if len(sys.argv) > 1:
        main()
    else:
        # Otherwise, run in interactive mode
        interactive_mode()

# ====================================================================
# USAGE EXAMPLES:
# ====================================================================

"""
Command line usage examples:

1. Extract all frames:
   python video_to_frames.py video.mp4

2. Extract every 10th frame:
   python video_to_frames.py video.mp4 -i 10

3. Extract to specific directory:
   python video_to_frames.py video.mp4 -o my_frames

4. Extract frames 100-500 as JPG:
   python video_to_frames.py video.mp4 -s 100 -e 500 -f jpg

5. Extract for calibration (every 30th frame):
   python video_to_frames.py calibration_video.mp4 -i 30 -o calibration_images

Python script usage examples:

# Extract all frames
extract_all_frames("my_video.mp4", "output_frames")

# Extract every 10th frame
extract_every_nth_frame("my_video.mp4", n=10, output_dir="sparse_frames")

# Extract frames from 10-60 seconds
extract_frames_time_range("my_video.mp4", start_seconds=10, end_seconds=60)

# Extract frames for calibration
extract_frames_for_calibration("calibration_video.mp4", "calibration_images", interval=30)
"""