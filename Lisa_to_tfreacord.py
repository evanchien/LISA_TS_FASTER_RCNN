

import os, sys
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
sys.path.append('/home/evanchien/TF_model/models/research/')
from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

DATA_PATH = "/home/evanchien/lisa/data/"

df = pd.DataFrame(pd.read_csv(DATA_PATH+'/allAnnotations.csv',sep=',|;', header=0, engine='python'))
item_cnt,_ = df.shape

annotation = []
for x in df['Annotation tag']:
    if x not in annotation:
        annotation.append(x)


annotation_dict = {k:v for v, k in enumerate(annotation)}

def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_image_binary(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    image = Image.open(filename)
    image = np.asarray(image, np.uint8)
    # shape = np.array(image.shape, np.int32)
    return image.tobytes() # convert image to raw data bytes in the array.

def _validate_text(text):
    """If text is not str or unicode, then try to convert it to str."""

    if isinstance(text, str):
        return text
    elif isinstance(text, unicode):
        return text.encode('utf8', 'ignore')
    else:
        return str(text)

def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    # examples=[]
    current_file = df['Filename'][0]
    image_format ='png'.encode()
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for i in range(item_cnt):  
        file = df['Filename'][i]
        if file != current_file:
            # print(current_file)
            filename = (DATA_PATH+current_file).encode()
            # filename = file.encode()
            tf_example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(filename),
          'image/source_id': dataset_util.bytes_feature(filename),
          'image/encoded': dataset_util.bytes_feature(encoded_image_data),
          'image/format': dataset_util.bytes_feature(image_format),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
            }))
            
            writer.write(tf_example.SerializeToString())


            current_file = file
            xmins = []
            xmaxs = []
            ymins = []
            ymaxs = []
            classes_text = []
            classes = []
        filename = os.path.join(DATA_PATH, current_file)
        # encoded_image_data = get_image_binary(filename)
        with tf.gfile.GFile(filename, 'rb') as fid:
            encoded_image_data = fid.read() # Encoded image bytes
        height, width = cv2.imread(filename).shape[:2]
        xmins.append(df['Upper left corner X'][i]*1.0/width)
        xmaxs.append(df['Lower right corner X'][i]*1.0/width)
        ymins.append(df['Upper left corner Y'][i]*1.0/height)
        ymaxs.append(df['Lower right corner Y'][i]*1.0/height)
        classes_text.append(df['Annotation tag'][i].encode())
        classes.append(annotation_dict[df['Annotation tag'][i]]+1)
    writer.close()


if __name__ == '__main__':
  tf.app.run()