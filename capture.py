import freenect
import numpy as np
import cv2

def get_depth():
    depth, _ = freenect.sync_get_depth()
    # or uncomment the following to only capture IR frames
    # depth, _ = freenect.sync_get_video(format=freenect.VIDEO_IR_8BIT)
    depth = depth.astype(np.uint8)
    return depth

def main():
    i = 0
    while True:
        depth_frame = get_depth()
        cv2.imshow('depth frame', depth_frame)
        # 'q' to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        # uncomment the following line to save the images
        #imwrite('depth_%05.png' % i, depth_frame)
        i += 1
    cv2.destroyAllWindows()
    freenect.sync_stop()

if __name__ == "__main__":
    main()
