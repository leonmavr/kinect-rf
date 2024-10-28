import cv2
import os
import numpy as np

### Instructions
# 1. Hold left click to draw a bounding box around the head, THEN the hands
# 2. Press 'n' to jump to the next image or 'q' to quit
# 3. Results will be saved in directory `labelled`

drawing = False  # true if the mouse is pressed
ix, iy = -1, -1  # initial position of the mouse
boxes = []  # store bounding boxes
lbl_head = 1
lbl_hands = 2
win_title = 'first the head, then hands'


def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(win_title, img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        boxes.append((ix, iy, x, y))
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow(win_title, img)


def annotate_images(image_folder, output_file):
    global img

    with open(output_file, 'w') as f:
        for image_name in sorted(os.listdir(image_folder)):
            image_path = os.path.join(image_folder, image_name)
            img = cv2.imread(image_path)

            if img is None:
                continue
            img_height, img_width = img.shape[:2]
            label_image = np.zeros(img.shape[:2], np.uint8)
            print(label_image.shape)

            cv2.imshow(win_title, img)
            cv2.setMouseCallback(win_title, draw_bbox)
            print(f"Annotating {image_name}... Press 'n' to move to the next image or 'q' to quit.")
            key = cv2.waitKey(0) & 0xFF

            for i, box in enumerate(boxes):
                # first the head, then the hands:
                if i == 0:
                    label_val = lbl_head 
                else:
                    label_val = lbl_hands
                x0, y0, x1, y1 = box
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                w = abs(x1 - x0)
                h = abs(y1 - y0)
                label_image[y0:y0+h, x0:x0+w] = label_val
            # reset for the next image
            boxes.clear()
            cv2.imwrite('./labelled/%s' % image_name.replace('depth', 'lbl'), label_image)
            if key == ord('q'):
                break
            elif key == ord('n'):
                continue
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_folder = './depth/'
    output_file = 'bounding_boxes.txt'
    annotate_images(image_folder, output_file)
