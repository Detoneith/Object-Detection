# Object Detection

This is a simple Python application that demonstrates object detection using YOLO (You Only Look Once) with a graphical user interface built using Tkinter.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10.x: You can download it from [python.org](https://www.python.org/downloads/).
- OpenCV: Install using `pip install opencv-python`.
- TtkBootstrap: Install using `pip install ttkbootstrap`.

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/Detoneith/Objects-detection.git

2. Download the YOLO.weights, yolov3.cfg, and COCO.names files from the official [YOLO website](https://github.com/AlexeyAB/darknet) and place them in the project directory.

3. Run the application: python main.py

4. The graphical user interface will appear. Click the "Start Detection" button to begin object detection. If no camera is found, an error message will be displayed.

5. Detected objects will be highlighted in the video feed, and their labels will be displayed.

6. To stop the detection, click the "Stop Detection" button.

![Screenshot](screenshot/Capture.PNG)


## Contributing
Contributions are welcome! If you have any ideas, improvements, or issues, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the ![LICENSE](LICENSE) file for details.
