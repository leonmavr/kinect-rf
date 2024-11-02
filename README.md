# kinect-rf

## 1. About

In this repository, I develop a random forest classifier that recognises the
head and hands in depth images captured from Kinect v1.

<img src="https://raw.githubusercontent.com/leonmavr/kinect-rf/refs/heads/master/assets/kinect-v1.png" alt="Kinect v1" height="100">

## 2. Requirements

### 2.1 Python Packages

Install the required packages:

```
pip install -r requirements.txt
```

### 2.2 Kernel Permissions

You will need the `libfreenect` library, so find out how to install it for
distribution.

Before connecting your Kinect, you will need to change the kernel's permission
for the device:

`sudo vi /etc/udev/rules.d/60-libfreenect.rules `

And the paste the following:

```
# ATTR{product}=="Xbox NUI Motor" permissions
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02b0", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ad", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02ae", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02c2", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02be", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="02bf", MODE="0666"
```

Then, reload the `udev` permissions:

```
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### 2.3 (Optional) Capture frames

Connect your Kinect via the USB. The green frontal LED should flash a few
times. Follow [yasupi's](https://github.com/OpenKinect/libfreenect/issues/580#issuecomment-643440616)
workaround to get it running and capture images by running:
`freenect-micview` on one terminal, then  
`freenect-camtest` optionally to test the camera - close it if it works, then  

Then, to view the frames execute:

`freenect-glview` on another terminal.

Or if you want to captre and save the frames as greyscale images, run my
`capture.py` script and make sure to uncomment the `imwrite` line:

```
python capture.py
```

**NOTE**: Capturing frames is optional. I have stored some pre-recorded frames
in `depth_train.zip`.

## 3. Training

### 3.1 Pre-trained classifiers

This repository contains some serialised (via `pickle`) head and hand
classifiers in the `clf` directory. If you want to train your own, follow
section 3.2, otherwise skip directly to 3.3.

### 3.2. Training a Head and Hand Classifier

Your training data must be stored as greyscale images in directory
`depth_train`. I have pre-recorded and zipped some data, so if you wish to use
it do:

```
unzip depth_train.zip
```

If you still need more pre-recorded data, you can extract the frames of the
testing video and select some for training:

```
mkdir temp
ffmpeg -i test_videos/2024_09_30.mp4 -vf fps=1 temp/depth_%05d.png
```

You can select as many frames as you like and add them to the `depth_train`
directory.

Next, you can annotate the training data:

```
python annot.py
```

In this script, draw a bounding box around the head and one around each hand,
keeping them tight. ALWAYS draw the one around the head first.

Now for each annotated depth image, you will have one labelled one in directory
`labelled`. Training can begin, so run:

```
python train_rf.py
```

### 3.3. Running the Demo and Visualising the Predictions

You should have exported your classifier as a pickled file. If you don't want to
use the default one, just edit the following line in `demo.py`:

```python
clf_path = os.path.join('clf', 'rf_head_hands_02.clf')
```

Then you can run the demo:

```
python demo.py
```

This will perform classification and draw a blue bounding box around the head
and two green ones around the hands.

<img src="https://raw.githubusercontent.com/leonmavr/kinect-rf/refs/heads/master/assets/demo_screenshot.png" alt="demo screenshot" height="230">
