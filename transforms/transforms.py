
import random
import math
import numbers

import cv2
import numpy as np

import torch

class Compose:
    """Composes several transforms together.

    Args:
        transforms(list of 'Transform' object): list of transforms to compose

    """    

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img):

        for trans in self.transforms:
            img = trans(img)
        
        return img
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


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
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped (w / h)
        interpolation: Default: cv2.INTER_LINEAR: 
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation='linear'):

        self.methods={
            "area":cv2.INTER_AREA, 
            "nearest":cv2.INTER_NEAREST, 
            "linear" : cv2.INTER_LINEAR, 
            "cubic" : cv2.INTER_CUBIC, 
            "lanczos4" : cv2.INTER_LANCZOS4
        }

        self.size = (size, size)
        self.interpolation = self.methods[interpolation]
        self.scale = scale
        self.ratio = ratio

    
    def __call__(self, img):
        h, w, _ = img.shape

        area = w * h

        for attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            target_ratio = random.uniform(*self.ratio) 

            output_h = int(round(math.sqrt(target_area * target_ratio)))
            output_w = int(round(math.sqrt(target_area / target_ratio))) 

            if random.random() < 0.5:
                output_w, output_h = output_h, output_w 

            if output_w <= w and output_h <= h:
                topleft_x = random.randint(0, w - output_w)
                topleft_y = random.randint(0, h - output_h)
                break

        if output_w > w or output_h > h:
            output_w = min(w, h)
            output_h = output_w
            topleft_x = random.randint(0, w - output_w) 
            topleft_y = random.randint(0, h - output_w)

        cropped = img[topleft_y : topleft_y + output_h, topleft_x : topleft_x + output_w]

        resized = cv2.resize(cropped, self.size, interpolation=self.interpolation)

        return resized
    
    def __repr__(self):
        for name, inter in self.methods.items():
            if inter == self.interpolation:
                inter_name = name

        interpolate_str = inter_name
        format_str = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_str += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_str += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_str += ', interpolation={0})'.format(interpolate_str)

        return format_str


class RandomHorizontalFlip:
    """Horizontally flip the given opencv image with given probability p.

    Args:
        p: probability of the image being flipped
    """
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img):
        """
        Args:
            the image to be flipped
        Returns:
            flipped image
        """
        if random.random() < self.p:
            img = cv2.flip(img, 1)
        
        return img


class ColorJitter:
    """Scale hue, saturation, and brightness with coefficients uniformly
    drawn from [0.6, 1.4]

    Args:
        h: range to scale hue
        b: range to scale brightness
        s: range to scale saturation
    """

    def __init__(self, h=[0.6, 1.4], b=[0.6, 1.4], s=[0.6, 1.4]):
        self.h = h
        self.b = b
        self.s = s

    def __call__(self, img):
        """
        Args:
            img to be jittered
        Returns:
            jittered img
        """

        h_factor = random.uniform(*self.h)
        b_factor = random.uniform(*self.b)
        s_factor = random.uniform(*self.s)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, b = cv2.split(img_hsv)

        h = h.astype('float32')
        s = s.astype('float32')
        b = b.astype('float32')

        h = h * h_factor
        b = b * b_factor
        s = s * s_factor

        print(h_factor, b_factor, s_factor)
        #normalize
        h = h / np.max(h) * 255.0
        print('h: ', np.max(h))
        b = b / np.max(b) * 255.0
        print('b: ', np.max(b))
        s = s / np.max(s) * 255.0
        print('s: ', np.max(s))

        #convert data type
        h = h.astype(img_hsv.dtype)
        b = b.astype(img_hsv.dtype)
        s = s.astype(img_hsv.dtype)

        img_hsv = cv2.merge((h, s, b))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        return img

class ToTensor:
    """convert an opencv image (h, w, c) ndarray to a pytorch float tensor 
    (c, h, w) 
    """

    def __call__(self, img):
        """
        Args:
            a numpy array (h, w, c) range from [0, 255]
        
        Returns:
            a pytorch tensor
        """
        #convert format H W C to C H W
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img.float()

        return img


class Normalize:
    """Normalize a numpy array (H, W, BGR order) with mean and standard deviation
    to float32 data type range from [0, 1]

    for each channel in numpy array:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """        
        dtype = img.dtype

        img = img.astype('float32')
        img = img / 255.0

        for index, mean in enumerate(self.mean):
            img[:, :, index] -= mean
        
        for index, std in enumerate(self.std):
            img[:, :, index] /= std
        
        img = img.astype(dtype)
        return img

        
class CenterCrop:
    """resize each image’s shorter edge to r pixels while keeping its aspect ratio. 
    Next, we crop out the cropped region in the center 
    Args:
        resized: resize image' shorter edge to resized pixels while keeping the aspect ratio
        cropped: output image size(h, w), if cropped is an int, then output cropped * cropped size
                 image
    """

    def __init__(self, resized=256, cropped=224, interpolation='linear'):

        methods = {
            "area":cv2.INTER_AREA, 
            "nearest":cv2.INTER_NEAREST, 
            "linear" : cv2.INTER_LINEAR, 
            "cubic" : cv2.INTER_CUBIC, 
            "lanczos4" : cv2.INTER_LANCZOS4
        }
        self.interpolation = methods[interpolation]

        self.resized = resized

        if isinstance(cropped, numbers.Number):
            cropped = (cropped, cropped)
        
        self.cropped = cropped

    def __call__(self, img):

        shorter = min(*img.shape[:2])

        scaler = float(self.resized) / shorter

        img = cv2.resize(img, (0, 0), fx=scaler, fy=scaler, interpolation=self.interpolation)

        h, w, _ = img.shape

        topleft_x = int((w - self.cropped[1]) / 2)
        topleft_y = int((h - self.cropped[0]) / 2)

        center_cropped = img[topleft_y : topleft_y + self.cropped[0], 
                             topleft_x : topleft_x + self.cropped[1]]

        return center_cropped
