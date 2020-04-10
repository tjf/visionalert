import os

import numpy
import pytest

import visionalert.tensorflow as tf


@pytest.fixture
def label_file(tmpdir):
    fh = tmpdir.join("labelmap.txt")
    fh.write(
        """???
cat
dog
person
"""
    )
    return os.path.join(fh.dirname, fh.basename)


@pytest.fixture
def mock_input_image():
    return numpy.zeros((480, 640, 3), numpy.uint8)


@pytest.fixture
def mock_interpreter(mocker):
    mocker.patch("visionalert.tensorflow.Interpreter")
    return tf.Interpreter.return_value


def test_calc_relative_box(mock_input_image):
    input_box = [0.2, 0.1, 0.8, 0.9]
    expected = tf.Rectangle(64, 96, 576, 384)
    result = tf.calc_relative_box(mock_input_image, input_box)
    assert result == expected


def test_calc_relative_box_with_overflows(mock_input_image):
    input_box = [-0.2, -0.1, 1.2, 1.2]
    expected = tf.Rectangle(1, 1, 640, 480)
    result = tf.calc_relative_box(mock_input_image, input_box)
    assert result == expected


def test_load_labels_and_correct_error(label_file):
    result = tf.load_labels(label_file)
    assert len(result) == 3
    assert result[0] != "???"


def test_objectdetector_init_tensorflow(label_file, mock_interpreter):
    tf.ObjectDetector("", label_file)
    tf.Interpreter.assert_called_once_with(model_path="")
    mock_interpreter.allocate_tensors.assert_called_once()


def test_objectdetector_pass_correct_size_image(
    label_file, mock_interpreter, mock_input_image
):
    detector = tf.ObjectDetector("", label_file)
    detector.detect_objects(mock_input_image)
    input_image = mock_interpreter.set_tensor.call_args[0][1]
    # Tensorflow expects a 1x300x300x3 input image shape
    assert input_image.shape == (1, 300, 300, 3)


def test_objectdetector_parse_detected_objects(
    label_file, mock_interpreter, mock_input_image
):
    detector = tf.ObjectDetector("", label_file)

    # Mock TensorFlow response -- Bounding box, label index, confidence score
    mock_interpreter_response = (([[0.2, 0.1, 0.8, 0.9]],), ([0],), ([0.8],))
    tf.Interpreter.return_value.get_tensor.side_effect = [
        mock_interpreter_response[0],
        mock_interpreter_response[1],
        mock_interpreter_response[2],
    ]

    expected = tf.Object("cat", 0.8, tf.Rectangle(64, 96, 576, 384))
    result = detector.detect_objects(mock_input_image)

    mock_interpreter.invoke.assert_called_once()
    assert result[0] == expected
