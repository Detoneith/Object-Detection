from Class.model import ObjectDetectionModel

class ObjectDetectionController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.start_button['command'] = self.start_detection
        self.view.stop_button['command'] = self.stop_detection

    def start_detection(self):
        self.model = ObjectDetectionModel('yolov3.weights', 'yolov3.cfg', 'coco.names')
        self.view.start_button['state'] = 'disabled'
        self.view.stop_button['state'] = 'normal'
        self.update_display_image()  # Call the method to update the display

    def update_display_image(self):
        imgtk = self.model.update_frame()  # Update the image in the model
        self.view.display_label.imgtk = imgtk
        self.view.display_label.configure(image=imgtk)
        if self.model.running:
            self.view.after(10, self.update_display_image)  # Update the image periodically

    def stop_detection(self):
        self.model.stop()
        self.view.start_button['state'] = 'normal'
        self.view.stop_button['state'] = 'disabled'
        self.model.clear_label_image()
        self.view.clear_display_image()

