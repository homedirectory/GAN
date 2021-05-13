import tensorflow as tf
import os
import numpy as np

def load_image(path, dtype):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype)
    # normalize to [-1, 1]
    image = (image - 0.5) * 2
    
    return image


def read_dir(dir_path, string=""):
    paths = []

    with os.scandir(dir_path) as it:
        for file in it:
            if len(string) == 0 or file.path.endswith(string):
                paths.append(file.path)

    return paths
    
    
def load_npy(path, dtype='float16'):
    data = np.load(path).astype(dtype)
    
    return data
