import numpy as np
import cv2 as cv
import glob
import os

# ====================================================================
# STEP 1: PREPARE YOUR ENVIRONMENT AND CHECKERBOARD
# ====================================================================

"""
Before running this code:
1. Print a checkerboard pattern (7x6 internal corners = 8x7 squares)
2. Mount it on rigid, flat surface (cardboard, clipboard, etc.)
3. Take 15-20 photos with your smartphone from different angles/distances
4. Save all photos in a folder (e.g., 'calibration_images/')
5. Make sure photos are clear and checkerboard is fully visible
"""

# ====================================================================
# STEP 2: DEFINE CALIBRATION PARAMETERS
# ====================================================================

# Checkerboard dimensions (internal corners, not squares!)
CHECKERBOARD_SIZE = (9, 6)  # 7 corners horizontally, 6 vertically
SQUARE_SIZE = 2.2  # Size of each square (we'll use 1.0 as unit, you can change to mm)

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ====================================================================
# STEP 3: PREPARE OBJECT POINTS (3D WORLD COORDINATES)
# ====================================================================

# Create 3D points for the checkerboard corners in world space
# We assume the checkerboard is on Z=0 plane
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE  # Scale by square size

print(f"Object points shape: {objp.shape}")
print(f"First few object points:\n{objp[:5]}")

# ====================================================================
# STEP 4: ARRAYS TO STORE CALIBRATION DATA
# ====================================================================

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane
successful_images = []  # Keep track of which images worked

# ====================================================================
# STEP 5: LOAD AND PROCESS CALIBRATION IMAGES
# ====================================================================

# Change this path to where your calibration images are stored
image_path = '/workspace/repo/calibration_images/*.jpg'  # Adjust extension as needed
images = glob.glob(image_path)

if not images:
    print(f"No images found at {image_path}")
    print("Please check your image path and file extensions")
    exit()

print(f"Found {len(images)} images")

for i, fname in enumerate(images):
    print(f"Processing image {i+1}/{len(images)}: {os.path.basename(fname)}")
    
    # Load image
    img = cv.imread(fname)
    if img is None:
        print(f"  Could not load image: {fname}")
        continue
    
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)
    
    if ret:
        print(f"  ✓ Found checkerboard corners")
        
        # Add object points (same for each image)
        objpoints.append(objp)
        
        # Refine corner positions to sub-pixel accuracy
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        successful_images.append(fname)
        
        # Optional: Draw and display corners for verification
        img_with_corners = img.copy()
        cv.drawChessboardCorners(img_with_corners, CHECKERBOARD_SIZE, corners2, ret)
        
        # Uncomment these lines if you want to see each image with corners
        # cv.imshow('Checkerboard Corners', img_with_corners)
        # cv.waitKey(500)  # Display for 500ms
        
    else:
        print(f"  ✗ Could not find checkerboard corners")

# cv.destroyAllWindows()

print(f"\nSuccessfully processed {len(objpoints)} out of {len(images)} images")

if len(objpoints) < 10:
    print("Warning: Less than 10 successful images. Consider taking more calibration photos.")

# ====================================================================
# STEP 6: CAMERA CALIBRATION
# ====================================================================

if len(objpoints) > 0:
    print("\nPerforming camera calibration...")
    
    # Get image dimensions from the first successful image
    img = cv.imread(successful_images[0])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # (width, height)
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    print(f"Calibration successful: {ret}")
    print(f"RMS reprojection error: {ret:.4f} pixels")
    
    # ====================================================================
    # STEP 7: DISPLAY CALIBRATION RESULTS
    # ====================================================================
    
    print("\n" + "="*50)
    print("CALIBRATION RESULTS")
    print("="*50)
    
    print(f"\nCamera Matrix (Intrinsic Parameters):")
    print(camera_matrix)
    print(f"\nFocal Length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
    print(f"Principal Point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")
    
    print(f"\nDistortion Coefficients:")
    print(dist_coeffs)
    print(f"k1={dist_coeffs[0][0]:.6f}, k2={dist_coeffs[0][1]:.6f}, p1={dist_coeffs[0][2]:.6f}, p2={dist_coeffs[0][3]:.6f}, k3={dist_coeffs[0][4]:.6f}")
    
    # ====================================================================
    # STEP 8: CALCULATE REPROJECTION ERROR
    # ====================================================================
    
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    print(f"\nMean reprojection error: {mean_error:.4f} pixels")
    
    if mean_error < 1.0:
        print("✓ Excellent calibration (error < 1.0 pixel)")
    elif mean_error < 2.0:
        print("✓ Good calibration (error < 2.0 pixels)")
    else:
        print("⚠ Consider taking more/better calibration images (error > 2.0 pixels)")
    
    # ====================================================================
    # STEP 9: SAVE CALIBRATION DATA
    # ====================================================================
    
    # Save calibration results
    np.savez('camera_calibration.npz',
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             rvecs=rvecs,
             tvecs=tvecs,
             mean_error=mean_error,
             image_size=image_size)
    
    print(f"\nCalibration data saved to 'camera_calibration.npz'")
    
    # ====================================================================
    # STEP 10: TEST UNDISTORTION ON A SAMPLE IMAGE
    # ====================================================================
    
    print("\nTesting undistortion on sample image...")
    
    # Load a test image (using first successful calibration image)
    test_img = cv.imread(successful_images[0])
    h, w = test_img.shape[:2]
    
    # Get optimal new camera matrix
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    
    # Method 1: Using cv.undistort()
    dst = cv.undistort(test_img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop the image using ROI
    x, y, w_roi, h_roi = roi
    dst_cropped = dst[y:y+h_roi, x:x+w_roi]
    
    # Save results
    cv.imwrite('original_distorted.jpg', test_img)
    cv.imwrite('undistorted_full.jpg', dst)
    cv.imwrite('undistorted_cropped.jpg', dst_cropped)
    
    print("Sample undistortion results saved:")
    print("  - original_distorted.jpg")
    print("  - undistorted_full.jpg") 
    print("  - undistorted_cropped.jpg")
    
    # ====================================================================
    # STEP 11: SHOW HOW TO USE CALIBRATION DATA LATER
    # ====================================================================
    
    print("\n" + "="*50)
    print("HOW TO USE CALIBRATION DATA IN YOUR PROJECT")
    print("="*50)
    
    print("""
# To load calibration data later:
calibration_data = np.load('camera_calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# To undistort new images:
def undistort_image(img, camera_matrix, dist_coeffs):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    dst = cv.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

# Example usage:
# new_img = cv.imread('your_image.jpg')
# undistorted_img = undistort_image(new_img, camera_matrix, dist_coeffs)
""")

else:
    print("No successful calibration images found. Please check your images and try again.")

# ====================================================================
# ADDITIONAL HELPER FUNCTIONS
# ====================================================================

def load_calibration_data(filename='camera_calibration.npz'):
    """Load previously saved calibration data"""
    try:
        data = np.load(filename)
        return data['camera_matrix'], data['dist_coeffs']
    except FileNotFoundError:
        print(f"Calibration file {filename} not found!")
        return None, None

def undistort_image(img, camera_matrix, dist_coeffs, crop=True):
    """Undistort an image using calibration parameters"""
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    dst = cv.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    
    if crop:
        x, y, w_roi, h_roi = roi
        dst = dst[y:y+h_roi, x:x+w_roi]
    
    return dst

# Example of how to use the calibration on new images:
def process_dataset_with_calibration():
    """Example function showing how to apply calibration to your dataset"""
    
    # Load calibration data
    camera_matrix, dist_coeffs = load_calibration_data()
    
    if camera_matrix is None:
        print("Please run calibration first!")
        return
    
    # Process your dataset images
    dataset_images = glob.glob('dataset_images/*.jpg')  # Change path as needed
    
    for img_path in dataset_images:
        # Load image
        img = cv.imread(img_path)
        
        # Undistort
        undistorted = undistort_image(img, camera_matrix, dist_coeffs)
        
        # Save undistorted image
        output_path = img_path.replace('.jpg', '_undistorted.jpg')
        cv.imwrite(output_path, undistorted)
        
        print(f"Processed: {img_path}")

print("\nCalibration script completed!")
print("Run process_dataset_with_calibration() to apply calibration to your dataset images.")