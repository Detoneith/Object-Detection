import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from Class.model import ObjectDetectionModel
from Class.view import ObjectDetectionView
from Class.controller import ObjectDetectionController

def main():
    app = ttk.Window(title="Object Detection", themename="cyborg", size=[800, 600], resizable=[False,False])
    
    model = ObjectDetectionModel('yolov3.weights', 'yolov3.cfg', 'coco.names')
    view = ObjectDetectionView(app, model)
    controller = ObjectDetectionController(model, view)

    app.mainloop()

if __name__ == "__main__":
    main()
