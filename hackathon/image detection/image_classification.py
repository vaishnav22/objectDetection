import os
import time
import tarfile
import glob
import six.moves.urllib as urllib
import cv2
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from utils.ssd_mobilenet_utils import *

def run_detection(image_data, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = sess.graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
           
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    detection_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    num_detections = sess.graph.get_tensor_by_name('num_detections:0')

    boxes, scores, classes, num = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                    feed_dict={image_tensor: image_data})
    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes).astype(np.int32)
    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)

    # Print predictions info
    #print('Found {} boxes.'.format(len(out_boxes)))
            
    return out_scores, out_boxes, out_classes

def image_object_detection(image_path, sess, colors):
    image = cv2.imread(image_path)

    image_data = preprocess_image(image, model_image_size=(300,300))
    out_scores, out_boxes, out_classes = run_detection(image_data, sess)

    # Draw bounding boxes on the image file
    image = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image_name = os.path.basename(image_path)
    print(image_name)
    cv2.imwrite(os.path.join("out/", "out_" + image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    output_image = cv2.imread(os.path.join("out/", "out_" + image_name))
    cv2.imshow("Output Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # What model to download
    model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    model_file = model_name + '.tar.gz'
    
    # Download model to model_data dir
    model_dir = 'model_data'
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    file_path = os.path.join(model_dir, model_file)

    # Load a (frozen) Tensorflow model into memory.
    path_to_ckpt = model_dir + '/' + model_name + '/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
    # label
    class_names = read_classes('model_data/coco_classes.txt')
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)

    # path = input("Enter the image path")
    image_path = "3.jpeg"
    
    with detection_graph.as_default():
        with tf.Session() as sess:
            '''
            # image_object_detection
            # Make a list of images
            images = glob.glob('./images/*.jpg')
            for fname in images:
                image_object_detection(fname, sess, colors)
            '''

            image_object_detection(image_path, sess, colors)
