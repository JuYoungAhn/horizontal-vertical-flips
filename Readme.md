# Image augmentation with vertical and horizontal flip 
* Manual data augmentation (no random) with vertical and horizontal flips using tensorflow image library.
* get_horizontal_vertical_flips method produces augmented tensor with vertical and horizontal flips. 
  * input: [sample, width, height, channel] 
  * output: [4*sample, width, height, channel]
* tensorflow's eager mode is available.