import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

def callback(msg):
    bridge = CvBridge()
    depth = bridge.imgmsg_to_cv2(msg, 'passthrough')
    center_y, center_x = depth.shape[0]//2, depth.shape[1]//2
    
    # Print value at center and nearby pixels
    print(f'Center pixel value: {depth[center_y, center_x]:.3f}m')
    for y in range(center_y-2, center_y+3, 2):
        for x in range(center_x-2, center_x+3, 2):
            print(f'Pixel at ({y},{x}): {depth[y,x]:.3f}m')
    
    valid = depth[(depth > 0) & np.isfinite(depth)]
    if len(valid) > 0:
        print(f'Min: {np.min(valid):.3f}m, Max: {np.max(valid):.3f}m')
    rospy.signal_shutdown('Done')

rospy.init_node('depth_checker')
rospy.Subscriber('/boxi/zed2i/depth/image_raw_uncompressed', Image, callback)
rospy.spin()
