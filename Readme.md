# Image augmentation with vertical and horizontal flip 
- KerasImageGenerator (https://keras.io/preprocessing/image/) with flip parameter produces randomly fliped image, so it was inconvenient method when randomness is not neccassary.
- This is manual data augmentation with vertical and horizontal flips using tensorflow.image library.
- In augmentation.py, get_horizontal_vertical_flips method produces augmented tensor with vertical and horizontal flips. 
        - input: [sample, width, height, channel] 
        - output: [4*sample, width, height, channel]