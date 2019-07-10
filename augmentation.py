import numpy as np
from tqdm import tqdm
import tensorflow as tf

def get_horizontal_vertical_flips(x, ch) : 
    """ get horizontal and vertical flips of images
    Args:
            x (ndarray): 4-D tensor 
            ch (int): feature channel
            - gray-scale image: 1
            - color image: 3
        Returns:
            ndarray: augmented image tensor (4-D)
    """
    SAMPLE = x.shape[0]
    WIDTH = x.shape[1]
    HEIGHT = x.shape[2]
      
    augmented_x = np.zeros(shape=[4*SAMPLE,WIDTH,HEIGHT,ch])
    for i in tqdm(range(0, SAMPLE)):
        fliped_tensor = tf.image.flip_left_right(x[i])
        fliped_image = tf.Session().run(fliped_tensor)
        fliped_tensor2 = tf.image.flip_up_down(x[i])
        fliped_image2 = tf.Session().run(fliped_tensor2)
        fliped_tensor3 = tf.image.flip_up_down(fliped_image)
        fliped_image3 = tf.Session().run(fliped_tensor3)
        
        augmented_x[4*i] = x[i] # original image
        augmented_x[4*i+1] = fliped_image # vertical fliped image
        augmented_x[4*i+2] = fliped_image2 # horizontal fliped image
        augmented_x[4*i+3] = fliped_image3 # vertical and horizontal fliped image
        
    return augmented_x