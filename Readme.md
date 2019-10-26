# Image augmentation with vertical and horizontal flip 
* Data augmentation with vertical and horizontal flips using tensorflow (v1.9.0).
* "get_horizontal_vertical_flips" produces augmented tensor with vertical and horizontal flips. 
  * input: [sample, width, height, channel] 
  * output: [4*sample, width, height, channel]
* The tensorflow's eager mode is available and recommended. 
