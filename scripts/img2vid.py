import cv2
import os
import argparse
from pathlib import Path
import re

def natural_sort_key(path):
    """
    Sort key function that extracts the numeric portion after the underscore.
    For filenames like '8980_15.jpg', it will sort based on the '15'.
    """
    # Extract the number after the underscore and before the extension
    match = re.search(r'_(\d+)\.jpg$', str(path))
    if match:
        return int(match.group(1))
    return 0

def convert_images_to_video(input_dir, output_path, fps=30):
    """
    Convert a directory of images to a video file.
    
    Args:
        input_dir (str): Directory containing the images
        output_path (str): Path for the output video file
        fps (int): Frames per second for the output video
    """
    # Convert to Path objects for better path handling
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get all jpg files and sort them using the natural sort key
    files = sorted([f for f in input_dir.glob("*.jpg")], key=natural_sort_key)
    if not files:
        raise ValueError(f"No JPG files found in {input_dir}")
    
    print(f"First few files in order: {[f.name for f in files[:5]]}")
    print(f"Total number of files: {len(files)}")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(files[0]))
    if first_image is None:
        raise ValueError(f"Failed to read first image: {files[0]}")
    
    height, width = first_image.shape[:2]
    
    # Try different codec options
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # More widely supported codec
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Fallback codec
    
    # Create video writer
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not video_writer.isOpened():
        raise ValueError("Failed to create VideoWriter")
    
    # Process all images
    for img_path in files:
        print(f"Processing {img_path.name}")
        img = cv2.imread(str(img_path))
        
        if img is not None:
            video_writer.write(img)
        else:
            print(f"Warning: Could not read {img_path}")
    
    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a sequence of images to video")
    parser.add_argument("-i", "--input", type=str, help="Directory containing input images", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output video file path", required=True)
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 30)")
    
    args = parser.parse_args()
    
    try:
        convert_images_to_video(args.input, args.output, args.fps)
    except Exception as e:
        print(f"Error: {e}")