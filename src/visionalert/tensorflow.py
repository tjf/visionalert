import logging
import platform

import cv2
import numpy
import tflite_runtime.interpreter as tflite

from visionalert.detection import Rectangle, DetectionResult

EDGETPU_SHARED_LIB = {
    "Linux": "libedgetpu.so.1",
    "Darwin": "libedgetpu.1.dylib",
    "Windows": "edgetpu.dll",
}[platform.system()]

logger = logging.getLogger(__name__)


def load_labels(filename):
    with open(filename, "r") as label_file:
        labels = [line.strip() for line in label_file.readlines()]
        # Label map with TF Lite has '???' on the first line that seems to mess things up
        if labels[0] == "???":
            del labels[0]
        return labels


def calc_bounding_box(frame, box):
    h, w, _ = frame.shape
    start_y = int(max(1, (box[0] * h)))
    start_x = int(max(1, (box[1] * w)))
    end_y = int(min(h, (box[2] * h)))
    end_x = int(min(w, (box[3] * w)))
    return Rectangle(start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y)


def create_detector(model, label, input_width=300, input_height=300):
    try:
        delegate = [tflite.load_delegate(EDGETPU_SHARED_LIB)]
        logger.info("Initialized EdgeTPU device.  (Sweet!)")
    except Exception as e:
        logger.info(f"Unable to initialize EdgeTPU, using CPU: {str(e)}")
        delegate = None

    interpreter = tflite.Interpreter(model_path=model, experimental_delegates=delegate)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    labels = load_labels(label)

    def detect_function(frame):
        input_frame = frame
        if frame.shape[1] != input_width and frame.shape[0] != input_height:
            input_frame = cv2.resize(frame, (input_width, input_height))

        interpreter.set_tensor(
            input_details[0]["index"], numpy.expand_dims(input_frame, axis=0)
        )

        interpreter.invoke()

        return [
            DetectionResult(
                name=labels[int(label_index)],
                confidence=score,
                coordinates=calc_bounding_box(frame, box),
            )
            for box, label_index, score in zip(
                interpreter.get_tensor(output_details[0]["index"])[0],
                interpreter.get_tensor(output_details[1]["index"])[0],
                interpreter.get_tensor(output_details[2]["index"])[0],
            )
        ]

    return detect_function
