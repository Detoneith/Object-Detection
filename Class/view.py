import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from Class.model import ObjectDetectionModel

class ObjectDetectionView(ttk.Frame):
    def __init__(self, master, model):
        super().__init__(master)
        self.master = master
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.model = model

        self.create_widgets()

    def create_widgets(self):
        self.display_label = ttk.Label(self.master)
        self.display_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.start_button = ttk.Button(self.master, text="Start Detection", command=self.start_detection)
        self.start_button.grid(row=1, column=0, pady=10, padx=10, sticky="w")
        
        self.stop_button = ttk.Button(self.master, text="Stop Detection", command=self.stop_detection, state="disabled")
        self.stop_button.grid(row=1, column=1, pady=10, padx=10, sticky="e")

    def start_detection(self):
        global detector
        detector = ObjectDetectionModel('yolov3.weights', 'yolov3.cfg', 'coco.names')
        self.update_display_image()

    def update_display_image(self):
        if self.model.imgtk is not None:
            self.display_label.imgtk = self.model.imgtk
            self.display_label.configure(image=self.model.imgtk)
            self.after(10, self.update_display_image)

    def stop_detection(self):
        global detector
        detector.stop()
        detector.clear_label_image()
        self.clear_display_image()

    def clear_display_image(self):
        self.display_label.config(image='')
