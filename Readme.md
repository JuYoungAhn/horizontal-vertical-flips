# Image augmentation with vertical and horizontal flip 
* This is manual data augmentation with vertical and horizontal flips using tensorflow.image library.
* In augmentation.py, get_horizontal_vertical_flips method produces augmented tensor with vertical and horizontal flips. 
  * input: [sample, width, height, channel] 
  * output: [4*sample, width, height, channel]
* If tensorflow's eager mode is enabled, it works much faster. 