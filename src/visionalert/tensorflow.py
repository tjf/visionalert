import cv2
import numpy
from tflite_runtime.interpreter import Interpreter

from visionalert import Rectangle, Object


def load_labels(filename):
    with open(filename, "r") as label_file:
        labels = [line.strip() for line in label_file.readlines()]
        # Label map with TF Lite has '???' on the first line that seems to mess things up
        if labels[0] == "???":
            del labels[0]
        return labels


def calc_relative_box(frame, box):
    h, w, _ = frame.shape
    start_y = int(max(1, (box[0] * h)))
    start_x = int(max(1, (box[1] * w)))
    end_y = int(min(h, (box[2] * h)))
    end_x = int(min(w, (box[3] * w)))
    return Rectangle(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)


class ObjectDetector:

    # These are static values for the MobileNet SSD model we're using
    MODEL_INPUT_HEIGHT = 300
    MODEL_INPUT_WIDTH = 300

    def __init__(self, model_file, label_file) -> None:
        self._interpreter = Interpreter(model_path=model_file)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        self._labels = load_labels(label_file)

    def detect_objects(self, frame):
        """
        Detect objects using the configured model

        :param frame: numby nd_array containing RGB24 pixels
        :return: list of Object namedtuples for each object detected containing
        bounding box and confidence score
        """
        h, w, _ = frame.shape
        original_frame = frame

        if w > self.MODEL_INPUT_WIDTH or h > self.MODEL_INPUT_HEIGHT:
            # Allegedly, we don't need to worry about aspect ratio skew
            frame = cv2.resize(frame, (self.MODEL_INPUT_WIDTH, self.MODEL_INPUT_HEIGHT))

        self._interpreter.set_tensor(
            self._input_details[0]["index"], numpy.expand_dims(frame, axis=0)
        )

        # Here be the magic!
        self._interpreter.invoke()

        boxes = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        names = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores = self._interpreter.get_tensor(self._output_details[2]["index"])[0]

        detection_results = []
        for i, score in enumerate(scores):
            box = calc_relative_box(original_frame, boxes[i])
            name = self._labels[int(names[i])]
            result = Object(name=name, confidence=score, coordinates=box)
            detection_results.append(result)

        return detection_results
