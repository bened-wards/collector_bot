import onnxruntime
import numpy as np
import cv2
import time

class YOLO:

    def __init__(self, model_path, confidence_threshold, iou_threshold):
        self.model_path = model_path

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.init_onnx()

    def init_onnx(self):
        self.session = onnxruntime.InferenceSession(self.model_path)

        model_inputs = self.session.get_inputs()
        self.input_names = [input.name for input in model_inputs]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print(self.input_height, self.input_width)

        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def detect(self, img):
        """Take an image and run inference to detect objects in the image

        Args:
            img (np.array): image from opencv camera object

        Returns:
            tuple(list, list, list): bounding boxes (xywh), confidence scores, class IDs
        """
        preprocessed_img = self.preprocess(img)

        results = self.session.run(self.output_names, {self.input_names[0]: preprocessed_img})

        self.boxes_xywh, self.scores, self.class_ids = self.postprocess(results)

        return self.boxes_xywh, self.scores, self.class_ids
    
    def preprocess(self, img):
        self.img_height, self.img_width = img.shape[:2]

        float_im = img.astype(np.float32)
        input_im = cv2.cvtColor(float_im, cv2.COLOR_BGR2RGB)

        input_im = cv2.resize(input_im, (self.input_width, self.input_height))

        input_im /= 255.0
        input_im = np.transpose(input_im, [2, 0, 1])
        input_im = np.expand_dims(input_im, 0)
        return input_im
    
    def postprocess(self, results):
        predictions = np.squeeze(results[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.confidence_threshold, :]
        scores = scores[scores > self.confidence_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes_xywh = self.extract_boxes(predictions)

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, self.confidence_threshold, self.iou_threshold)

        return boxes_xywh[indices], scores[indices], class_ids[indices]
    
    def extract_boxes(self, preds):
        # Extract boxes from predictions
        boxes_xywh = preds[:, :4]

        # Scale boxes to original image dimensions
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes_xywh = np.divide(boxes_xywh, input_shape, dtype=np.float32)
        boxes_xywh *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        return boxes_xywh
    
    def draw_boxes(self, img):
        if len(self.boxes_xywh) == 0:
            return img
        
        draw_scores = True
        alpha = 0.4
        annotated_img = img.copy()
        h, w = img.shape[:2]


        # Annotate image
        for bbox in self.boxes_xywh:
            x, y, w, h = bbox
            x_min = int(x - w / 2)
            y_min = int(y - h / 2)
            x_max = int(x + w / 2)
            y_max = int(y + h / 2)
            cv2.rectangle(annotated_img, (x_min, y_min), (x_max, y_max), (50, 255, 0), 2)  # Draw bounding box

        return annotated_img
    

    def draw_detections(self, img, draw_scores=True):
        if len(self.boxes_xywh) == 0:
            return img
        
        draw_scores = True
        annotated_img = img.copy()
        img_h, img_w = img.shape[:2]

        for class_id, box, score in zip(self.class_ids, self.boxes_xywh, self.scores):
            class_name = self.class_names[class_id]
            colour = self.colours[class_id]
            caption = f"{class_name}: {score:.2f}" if draw_scores else class_name

            self.draw_box(annotated_img, box, img_h, img_w, colour)
            self.draw_text(annotated_img, caption, box, colour)

        return annotated_img
    
    def draw_box(self, img, box, h, w, colour):
        x, y, w, h = box
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        x_max = int(x + w / 2)
        y_max = int(y + h / 2)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), colour, 2)

    def draw_text(self, img, text, box, colour):
        x, y, w, h = box
        x_min = int(x - w / 2)
        y_min = int(y - h / 2)
        cv2.putText(img, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=colour, thickness=2)


if __name__ == "__main__":
    # model_path = "models/tennisball_yolov8_320.onnx"
    model_path = "models/box4_yolov8n_640.onnx"
    confidence_threshold = 0.7
    iou_threshold = 0.5

    model = YOLO(model_path, confidence_threshold, iou_threshold)

    img = cv2.imread('models/tball.jpg')
    boxes, scores, class_ids = model.detect(img)

    camera = cv2.VideoCapture(2)

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"h={height},w={width}")

    sum_time = 0
    num_frames = 0
    while True:
        ret, img = camera.read()
        if not ret:
            break
        
        start_time = time.time()
        boxes_xywh, scores, class_ids = model.detect(img)
        if len(boxes_xywh) > 0:
            print(boxes_xywh[0])

        annotated_img = model.draw_boxes(img)
        end_time = time.time()
        sum_time += end_time - start_time
        num_frames += 1
        print(f"Time taken: {end_time - start_time:.2f}s")
        print(f"Average time: {sum_time / num_frames:.2f}")
        cv2.imshow('Annotated Image', annotated_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()







