import os
import argparse
import re
import subprocess
from pathlib import Path
from PIL import Image
import tempfile
import shutil

def natural_sort_key(path):
    """
    Sort key function for filenames with numeric patterns.
    """
    # Extract the numeric part from filenames like '000112.png'
    match = re.search(r'(\d+)\.png$', str(path))
    if match:
        return int(match.group(1))
    return 0

def convert_images_to_video(input_dir, output_path, fps=30):
    """
    Convert a directory of images to a video file using ffmpeg.
    
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
    
    # Get all png files and sort them using the natural sort key
    files = sorted([f for f in input_dir.glob("*.png")], key=natural_sort_key)
    if not files:
        raise ValueError(f"No PNG files found in {input_dir}")
    
    print(f"First few files in order: {[f.name for f in files[:5]]}")
    print(f"Total number of files: {len(files)}")
    
    # Create a temporary directory for preparation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        
        # Check the first image for dimensions and ensure they're even
        first_image = Image.open(files[0])
        width, height = first_image.size
        first_image.close()
        
        # Create a padded version of images if needed
        need_padding = width % 2 != 0 or height % 2 != 0
        
        # Create input file list for ffmpeg
        input_list_file = temp_dir_path / "input_list.txt"
        
        with open(input_list_file, 'w') as f:
            for i, img_path in enumerate(files):
                print(f"Processing {img_path.name}")
                
                # If we need padding, create a copy with even dimensions
                if need_padding:
                    try:
                        img = Image.open(img_path)
                        new_width = width + (1 if width % 2 != 0 else 0)
                        new_height = height + (1 if height % 2 != 0 else 0)
                        
                        # Create a new image with even dimensions
                        new_img = Image.new("RGB", (new_width, new_height))
                        new_img.paste(img, (0, 0))
                        
                        # Save to temp directory
                        temp_img_path = temp_dir_path / f"frame_{i:06d}.png"
                        new_img.save(temp_img_path)
                        f.write(f"file '{temp_img_path.absolute()}'\n")
                        f.write(f"duration {1/fps}\n") 
                    except Exception as e:
                        print(f"Warning: Could not process {img_path}: {e}")
                else:
                    # Use original image
                    f.write(f"file '{img_path.absolute()}'\n")
                    f.write(f"duration {1/fps}\n")
            
            # Add the last frame again without duration to avoid cutting off
            if files:
                if need_padding:
                    f.write(f"file '{(temp_dir_path / f'frame_{len(files)-1:06d}.png').absolute()}'\n")
                else:
                    f.write(f"file '{files[-1].absolute()}'\n")
        
        # Build the ffmpeg command
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',                   # Overwrite output file
            '-f', 'concat',         # Use concat format
            '-safe', '0',           # Allow absolute file paths
            '-i', str(input_list_file),  # Input file list
            '-vsync', 'vfr',        # Variable frame rate (helps with timestamp issues)
            '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
            str(output_path)        # Output file
        ]
        
        # Execute ffmpeg command
        try:
            print(f"Running ffmpeg command: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            print(f"Video saved to {output_path}")
        except subprocess.CalledProcessError as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            raise RuntimeError(f"ffmpeg error: {error_message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a sequence of images to video")
    parser.add_argument("-i", "--input", type=str, help="Directory containing input images", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output video file path", required=True)
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    
    args = parser.parse_args()
    
    try:
        convert_images_to_video(args.input, args.output, args.fps)
    except Exception as e:
        print(f"Error: {e}")