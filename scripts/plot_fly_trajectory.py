import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def extract_camera_positions_from_json(json_file_path):
    """
    Extract camera positions from the JSON file containing keyframes with Rt matrices.
    
    Args:
        json_file_path (str): Path to the JSON file
        
    Returns:
        tuple: (positions, orientations, frame_names) where positions is Nx3 array of camera positions
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    positions = []
    orientations = []
    frame_names = []
    
    # Extract camera positions from Rt matrices
    for keyframe in data['keyframes']:
        # Rt matrix is 4x4 transformation matrix
        rt_matrix = np.array(keyframe['Rt'])
        
        # Extract rotation (3x3) and translation (3x1) components
        R = rt_matrix[:3, :3]  # Rotation matrix
        t = rt_matrix[:3, 3]   # Translation vector
        
        # Camera position in world coordinates is -R^T * t
        camera_pos = -R.T @ t
        
        positions.append(camera_pos)
        orientations.append(R)
        frame_names.append(keyframe['info']['name'])
    
    return np.array(positions), orientations, frame_names

def plot_camera_trajectory(json_file_path, show_orientations=True, orientation_scale=0.5):
    """
    Plot the camera trajectory from the JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        show_orientations (bool): Whether to show camera orientations as arrows
        orientation_scale (float): Scale factor for orientation arrows
    """
    positions, orientations, frame_names = extract_camera_positions_from_json(json_file_path)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory path
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, alpha=0.7, label='Camera trajectory')
    
    # Plot camera positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='red', s=20, alpha=0.8, label='Camera positions')
    
    # Plot start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
               c='green', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
               c='orange', s=100, marker='s', label='End')
    
    # Optionally show camera orientations
    if show_orientations:
        # Show orientation every 10th frame to avoid clutter
        step = max(1, len(positions) // 20)
        for i in range(0, len(positions), step):
            pos = positions[i]
            R = orientations[i]
            
            # Camera's forward direction (negative z-axis in camera coordinates)
            forward = -R[:, 2] * orientation_scale
            
            # Draw arrow showing camera direction
            ax.quiver(pos[0], pos[1], pos[2], 
                     forward[0], forward[1], forward[2], 
                     color='purple', alpha=0.6, arrow_length_ratio=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory Visualization')
    ax.legend()
    
    # Make axes equal
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                         positions[:, 1].max() - positions[:, 1].min(),
                         positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"Total number of keyframes: {len(positions)}")
    print(f"Trajectory length: {np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f} units")
    print(f"Start position: ({positions[0, 0]:.2f}, {positions[0, 1]:.2f}, {positions[0, 2]:.2f})")
    print(f"End position: ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f}, {positions[-1, 2]:.2f})")

def plot_2d_trajectory(json_file_path):
    """
    Plot the camera trajectory in 2D (top-down view).
    
    Args:
        json_file_path (str): Path to the JSON file
    """
    positions, _, frame_names = extract_camera_positions_from_json(json_file_path)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top-down view (X-Z plane)
    ax1.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(positions[:, 0], positions[:, 2], c='red', s=10, alpha=0.8)
    ax1.scatter(positions[0, 0], positions[0, 2], c='green', s=100, marker='o', label='Start')
    ax1.scatter(positions[-1, 0], positions[-1, 2], c='orange', s=100, marker='s', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')
    ax1.set_title('Top-down view (X-Z plane)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Side view (X-Y plane)
    ax2.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax2.scatter(positions[:, 0], positions[:, 1], c='red', s=10, alpha=0.8)
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='orange', s=100, marker='s', label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Side view (X-Y plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    json_file_path = "metadata.json"  # Replace with your JSON file path
    
    try:
        # Plot 3D trajectory
        print("Plotting 3D camera trajectory...")
        plot_camera_trajectory(json_file_path, show_orientations=True)
        
        # Plot 2D trajectory views
        print("Plotting 2D trajectory views...")
        plot_2d_trajectory(json_file_path)
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file_path}'")
        print("Please make sure the JSON file exists in the current directory.")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check that the JSON file has the correct format.")