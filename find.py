"""
FIND WALDO IN AN IMAGE
"""
from matplotlib import pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import matplotlib
from PIL import Image
import matplotlib.patches as patches
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis
import argparse


MODEL_PATH = 'model/frozen_inference_graph.pb'
LABEL_PATH = "model/label.txt"
IMAGE_TO_ASSESS = "assets/eval/2.jpg"
OUT = "here"


def load_detection_graph():
    """
    Load a TensorFlow detection graph from a model file.

    This function creates a new TensorFlow computation graph and loads a pre-trained model into it.

    :return: A TensorFlow computation graph containing the loaded model.
    :rtype: tf.Graph
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_image_into_numpy_array(image):
    """
    Load a PIL Image into a NumPy array.

    This function takes a PIL Image as input, extracts its dimensions, and converts it into a NumPy array.

    :param image: PIL.Image.Image
        The input image to be converted.

    :return: A NumPy array representing the image with shape (height, width, channels).
    :rtype: np.ndarray
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run():
    """
    Run object detection on an image using a loaded detection graph and label maps.

    This function loads a detection graph, label maps, and an image to perform object detection.
    Detected objects are visualized on the image, and the result is saved to an output file.

    Dependencies:
        - TensorFlow (tf)
        - Pillow (PIL)
        - Matplotlib (plt)

    """
    detection_graph = load_detection_graph()
    label_map = label_map_util.load_labelmap(LABEL_PATH)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=1, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    with detection_graph.as_default():
        image_np = load_image_into_numpy_array(Image.open(IMAGE_TO_ASSESS))
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = tf.compat.v1.Session(graph=detection_graph).run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        if scores[0][0] < 0.1:
            print('Not found')

        vis.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=(12, 8))
        plt.imshow(image_np)
        plt.savefig(OUT)

if __name__ == "__main__":
    run()
