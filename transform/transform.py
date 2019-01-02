

import cv2


class ToFloat:
    """Convert an image data type to 32-bit floating point
    """

    def __call__(self, image):
        """
        Args:
            image (numpy array): Image to be converted to 32-bit floating point
        
        Returns:
            image (numpy array): Converted Image
        """
        return image.astype('float32')


class RandomResizedCrop:
    """Randomly crop a rectangle region whose aspect ratio is randomly sampled 
    in [3/4, 4/3] and area randomly sampled in [8%, 100%], then resize the cropped
    region into a 224-by-224 square image.

    Args:
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='linear'))

        methods=[("area", cv2.INTER_AREA), 
         ("nearest", cv2.INTER_NEAREST), 
         ("linear", cv2.INTER_LINEAR), 
         ("cubic", cv2.INTER_CUBIC), 
         ("lanczos4", cv2.INTER_LANCZOS4)]
        
        
        
