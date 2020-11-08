[comment]: <> (View In Markdown for best experience)
[comment]: <> (Install markdown-preview package for Atom)

## Slouch Detector

### Tasks

#### Analysis
- Hardcoded algorithm?
- Neural Network?

#### Image Processing
(Finding Data live)

#### UI/UX
Color/Angle Calibration
Alerts
Usability

#### Data Collection
(To develop algorithm)
- eye height
  - y value of center of eyes
- shoulder height
  - where shoulder connects to neck
- shoulder width
  - distance between the two points where shoulder angle changes to arm
- shoulder angle
  - angle given by current algorithm (after calibrating for sitting straight)
- slouching
  - 1 or 0 (yes or no)
- frame saved
  - 1 or 0 (yes or no)
  
- Any other data points?

##### How to collect data?
- run slouchdetector.py
- record frame shape
- calibrate (c) when you are sitting straight and see contours on shoulders
- pause (p) when you get the frame you need
- fill in [data sheet](https://docs.google.com/spreadsheets/d/1_Tx8jwl0R3LQXzvPiA0F6oislBNPiM2SeQxS4PeEdRE/edit?usp=sharing) for each frame
- save frames (s)
- unpause to get another frame


## General Notes
[Data Sheet](https://docs.google.com/spreadsheets/d/1_Tx8jwl0R3LQXzvPiA0F6oislBNPiM2SeQxS4PeEdRE/edit?usp=sharing)

Resources in Resources.txt
