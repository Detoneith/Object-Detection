import cv2
from PIL import Image, ImageTk
from ttkbootstrap.dialogs import Messagebox

class ObjectDetectionModel:
    def __init__(self, weights_path, config_path, names_path):
        self.model = cv2.dnn.readNet(weights_path, config_path)
        
        self.classes = []
        with open(names_path, 'r') as f:
            self.classes = f.read().splitlines()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            Messagebox.show_error("Erreur", "Camera not found.")
            return
        
        self.running = True
        self.imgtk = None

    def stop(self):
        self.running = False
        self.cap.release()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop()
            return
        
        frame = cv2.resize(frame, (640, 480))
        
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.model.setInput(blob)
        output_layers_names = self.model.getUnconnectedOutLayersNames()
        layer_outputs = self.model.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = scores.argmax()
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[i % len(colors)]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 5), font, 1, color, 2)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2image))
        return self.imgtk
    
    
    def clear_label_image(self):
        self.imgtk = None