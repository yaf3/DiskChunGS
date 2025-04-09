import numpy as np
import matplotlib.pyplot as plt
import pykitti
from scipy.ndimage import distance_transform_edt
import skimage.color 
import scipy.sparse
from scipy.sparse.linalg import spsolve  # Add this for the sparse solver
import torch
import time
import os
from PIL import Image

def generate_depth_from_lidar_odometry(dataset, frame_idx):
    """Generate a depth image by projecting Velodyne points onto the image plane.
    
    Args:
        dataset: pykitti odometry dataset object
        frame_idx: Frame index
        
    Returns:
        depth_img: Depth image
    """
    # Get velodyne points for the frame
    velo_points = dataset.get_velo(frame_idx)
    
    # Get camera image for reference
    cam_img = np.array(dataset.get_cam2(frame_idx))
    img_h, img_w, _ = cam_img.shape
    
    # Create empty depth map
    depth_map = np.zeros((img_h, img_w))
    
    # Get calibration matrices
    # In the odometry dataset, we need to use the correct transformations
    # The pykitti odometry module already computes T_cam2_velo for us
    T_cam2_velo = dataset.calib.T_cam2_velo
    
    # Get projection matrix
    P_rect = dataset.calib.P_rect_20  # Projection matrix for camera 2
    
    # Convert to homogeneous coordinates
    velo_points_h = np.hstack((velo_points[:, :3], np.ones((velo_points.shape[0], 1))))
    
    # Transform from velodyne coordinates to camera coordinates
    cam_points = np.dot(T_cam2_velo, velo_points_h.T).T
    
    # Filter points behind the camera
    cam_points = cam_points[cam_points[:, 2] > 0]
    
    # Project to image plane
    pts_2d = np.dot(P_rect[:3, :3], cam_points[:, :3].T).T + P_rect[:3, 3]
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    
    # Filter points outside the image
    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) & \
           (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
    pts_2d = pts_2d[mask]
    depths = cam_points[mask, 2]  # Z-coordinate is the depth
    
    # Convert to pixel coordinates
    pts_2d = pts_2d[:, :2].astype(np.int32)
    
    # Assign depth values to the depth map
    for i in range(pts_2d.shape[0]):
        x, y = pts_2d[i, 0], pts_2d[i, 1]
        # If multiple points project to the same pixel, keep the closest one
        if depth_map[y, x] == 0 or depths[i] < depth_map[y, x]:
            depth_map[y, x] = depths[i]
    
    return depth_map

def fill_depth_colorization(imgRgb=None, imgDepthInput=None, alpha=1):
	imgIsNoise = imgDepthInput == 0
	maxImgAbsDepth = np.max(imgDepthInput)
	imgDepth = imgDepthInput / maxImgAbsDepth
	imgDepth[imgDepth > 1] = 1
	(H, W) = imgDepth.shape
	numPix = H * W
	indsM = np.arange(numPix).reshape((W, H)).transpose()
	knownValMask = (imgIsNoise == False).astype(int)
	grayImg = skimage.color.rgb2gray(imgRgb)
	winRad = 1
	len_ = 0
	absImgNdx = 0
	len_window = (2 * winRad + 1) ** 2
	len_zeros = numPix * len_window

	cols = np.zeros(len_zeros) - 1
	rows = np.zeros(len_zeros) - 1
	vals = np.zeros(len_zeros) - 1
	gvals = np.zeros(len_window) - 1

	for j in range(W):
		for i in range(H):
			nWin = 0
			for ii in range(max(0, i - winRad), min(i + winRad + 1, H)):
				for jj in range(max(0, j - winRad), min(j + winRad + 1, W)):
					if ii == i and jj == j:
						continue

					rows[len_] = absImgNdx
					cols[len_] = indsM[ii, jj]
					gvals[nWin] = grayImg[ii, jj]

					len_ = len_ + 1
					nWin = nWin + 1

			curVal = grayImg[i, j]
			gvals[nWin] = curVal
			c_var = np.mean((gvals[:nWin + 1] - np.mean(gvals[:nWin+ 1])) ** 2)

			csig = c_var * 0.6
			mgv = np.min((gvals[:nWin] - curVal) ** 2)
			if csig < -mgv / np.log(0.01):
				csig = -mgv / np.log(0.01)

			if csig < 2e-06:
				csig = 2e-06

			gvals[:nWin] = np.exp(-(gvals[:nWin] - curVal) ** 2 / csig)
			gvals[:nWin] = gvals[:nWin] / sum(gvals[:nWin])
			vals[len_ - nWin:len_] = -gvals[:nWin]

	  		# Now the self-reference (along the diagonal).
			rows[len_] = absImgNdx
			cols[len_] = absImgNdx
			vals[len_] = 1  # sum(gvals(1:nWin))

			len_ = len_ + 1
			absImgNdx = absImgNdx + 1

	vals = vals[:len_]
	cols = cols[:len_]
	rows = rows[:len_]
	A = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	rows = np.arange(0, numPix)
	cols = np.arange(0, numPix)
	vals = (knownValMask * alpha).transpose().reshape(numPix)
	G = scipy.sparse.csr_matrix((vals, (rows, cols)), (numPix, numPix))

	A = A + G
	b = np.multiply(vals.reshape(numPix), imgDepth.flatten('F'))

	#print ('Solving system..')

	new_vals = spsolve(A, b)
	new_vals = np.reshape(new_vals, (H, W), 'F')

	#print ('Done.')

	denoisedDepthImg = new_vals * maxImgAbsDepth
    
	output = denoisedDepthImg.reshape((H, W)).astype('float32')

	output = np.multiply(output, (1-knownValMask)) + imgDepthInput
    
	return output

def colorize_depth(depth_map, max_depth=50):
    """Convert depth map to color image for visualization."""
    # Normalize depth map
    norm_depth = np.clip(depth_map, 0, max_depth) / max_depth
    
    # Apply colormap
    colored_depth = plt.cm.jet(norm_depth)
    
    # Convert to 0-255 range
    return (colored_depth[:, :, :3] * 255).astype(np.uint8)

# Example usage:
if __name__ == "__main__":
    # Change these paths to your KITTI data location
    basedir = '/data/kitti/data_odometry_color/dataset'
    sequence = '00'  # Use sequences 00-10 which have ground truth
    os.makedirs(f"{basedir}/sequences/{sequence}/depth", exist_ok=True)
    os.makedirs(f"{basedir}/sequences/{sequence}/depth_completed", exist_ok=True)
    
    # Load dataset
    dataset = pykitti.odometry(basedir, sequence)
    
    for frame_idx in range(len(dataset)):
        print(f"Processing frame {frame_idx}/{len(dataset)}")
        
        # if (frame_idx < 2454):
        #     continue
        
        # Get RGB image for reference
        rgb_img = np.array(dataset.get_cam2(frame_idx))
        
        # Generate depth from LiDAR
        depth_lidar = generate_depth_from_lidar_odometry(dataset, frame_idx)
        
        print("Filling gaps in LiDAR depth map...")
        # Fill small gaps in the LiDAR depth map
        
        depth_lidar_filled = fill_depth_colorization(rgb_img, depth_lidar)
        
        # color_depth_lidar = colorize_depth(depth_lidar)
        # color_depth_lidar_filled = colorize_depth(depth_lidar_filled)
        
        # # Visualize
        # fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # axes[0, 0].imshow(rgb_img)
        # axes[0, 0].set_title('RGB Image')
        # axes[0, 0].axis('off')
        
        # axes[0, 1].imshow(color_depth_lidar)
        # axes[0, 1].set_title('Depth from LiDAR (Raw)')
        # axes[0, 1].axis('off')
        
        # axes[1, 0].imshow(color_depth_lidar_filled)
        # axes[1, 0].set_title('Depth from LiDAR (Filled)')
        # axes[1, 0].axis('off')
        
        # plt.tight_layout()
        # plt.show()
        
        depth_scaled = (depth_lidar * 1000).astype(np.uint16)
        depth_image = Image.fromarray(depth_scaled)
        depth_image.save(f"{basedir}/sequences/{sequence}/depth/{frame_idx:06d}.png")
        
        depth_scaled_filled = (depth_lidar_filled * 1000).astype(np.uint16)
        depth_image_filled = Image.fromarray(depth_scaled_filled)
        depth_image_filled.save(f"{basedir}/sequences/{sequence}/depth_completed/{frame_idx:06d}.png")