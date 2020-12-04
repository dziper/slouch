# Slouch Detector
###### Daniel Ziper, Aakash Davasam, and Sylvia Adams
Our project uses the webcam on the user’s computer to monitor their posture and give notifications to sit up straight when it is detected that the user is slouching.

### Installation:
Please make sure you have the following packages installed before use:
- `opencv-python`
- `imutils`
- `numpy`
- `PyMsgBox`
- `scipy`
- `matplotlib`
- Packages included in standard Python install
    - `math`
    - `argparse`
    - `copy`
    - `csv`
    - `datetime`

You can use `pip` or `conda` to install these packages
- Ex. `pip3 install opencv-python`

### How to Use:
1. Run slouchDetector.py with Python 3.x (`python3 slouchDetector.py`) and make sure your face and shoulders are visible on the webcam.
2. First we need to grab the color of your shirt. Click ‘x’ so a still frame pops up and use your cursor to click and drag the portion of the screen so it creates a rectangle across your shoulders. Note that only your shirt color should be inside the green rectangle and nothing else. If you’re unhappy with the green rectangle, you can redo it by pressing ‘r’.
3. Once you have a good green rectangle, click ‘c’ to allow the program to crop the selected region so it can store it and do some quick maths.
4. When this is over, click ‘q’ so a new display shows up.
5. Now, using the sliders at the bottom of the window, adjust the yellow lines to fit your shoulders.
    - The top slider changes the outside length of the shoulder line. Move the bar to the left (smaller) or to the right (larger) until the end of the line is right at the end of your shoulder. You don’t want the points to curve down, but you also don’t want the green line to be too short.
    - The bottom slider changes the distance from your neck. If the green line starts too close to your neck, shift the bar to the right until the line is only on your shirt. This is especially useful if you are wearing a wide-neck or v-neck shirt.
6. Now we need to calibrate the algorithm. The mask should have your shirt/shoulders appearing white.  In order to calibrate, press ‘c’ when the green line matches the top of your shoulder. The better the selection, the better the detection will be. Calibration uses the past few seconds of data, so sit still for a few seconds before you press 'c'.
7. If, for any reason, you need to redo the rectangle cropping, you may do so by repeating steps 3 and 4. You can also adjust the spacing of the shoulders at any time by repeating step 6.
8. Now that you’ve set up the detector, it will continuously monitor your posture and display a red dot if it finds you are slouching. If your shoulders are not being detected correctly, feel free to calibrate color (by pressing 'x' and then cropping a portion of your shirt). You can also calibrate your posture again by pressing 'c'

#### Changing input source
If you have multiple webcams connected (ie a laptop camera and a USB camera) you may want to change your webcam source.
You can do this by adding the `-s` or `--source` runtime argument. The program will default to your laptop webcam `-s 0`.
To switch the source to your USB camera, when you run the program, run `python3 slouchDetector.py -s 1`
