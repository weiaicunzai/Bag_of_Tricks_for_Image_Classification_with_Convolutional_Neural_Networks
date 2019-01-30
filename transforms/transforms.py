
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


class ToCVImage:
    """Convert an Opencv image to a 3 channel uint8 image
    """

    def __call__(self, image):
        """
        Args:
            image (numpy array): Image to be converted to 32-bit floating point
        
        Returns:
            image (numpy array): Converted Image
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(iamge, cv2.COLOR_GRAY2BGR)
        
        image = image.astype('uint8')
            
        return image


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

    """Randomly change the brightness, contrast and saturation of an image

    Args:
        brightness: (float or tuple of float(min, max)): how much to jitter
            brightness, brightness_factor is choosen uniformly from[max(0, 1-brightness),
            1 + brightness] or the given [min, max], Should be non negative numbe
        contrast: same as brightness
        saturation: same as birghtness
        hue: same as brightness
    """        

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness)
        self.contrast = self._check_input(contrast)
        self.saturation = self._check_input(saturation)
        self.hue = self._check_input(hue)

    def _check_input(self, value):

        if isinstance(value, numbers.Number):
            assert value >= 0, 'value should be non negative'
            value = [max(0, 1 - value), 1 + value]
        
        elif isinstance(value, (list, tuple)):
            assert len(value) == 2, 'brightness should be a tuple/list with 2 elements'
            assert 0 <= value[0] <= value[1], 'max should be larger than or equal to min,\
            and both larger than 0'

        else:
            raise TypeError('need to pass int, float, list or tuple, instead got{}'.format(type(value).__name__))

        return value

    def __call__(self, img):
        """
        Args:
            img to be jittered
        Returns:
            jittered img
        """

        img_dtype = img.dtype
        h_factor = random.uniform(*self.hue)
        b_factor = random.uniform(*self.brightness)
        s_factor = random.uniform(*self.saturation)
        c_factor = random.uniform(*self.contrast)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img.astype('float32')

        #h
        img[:, :, 0] *= h_factor
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        #s
        img[:, :, 1] *= s_factor
        img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)

        #v
        img[:, :, 2] *= b_factor
        img[:, :, 2] = np.clip(img[:, :, 2], 0, 255)

        img = img.astype(img_dtype)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        #c
        img = img * c_factor
        img = img.astype(img_dtype)
        img = np.clip(img, 0, 255)

        return img

class ToTensor:
    """convert an opencv image (h, w, c) ndarray range from 0 to 255 to a pytorch 
    float tensor (c, h, w) ranged from 0 to 1
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
        img = img.float() / 255.0

        return img

class Normalize:
    """Normalize a torch tensor (H, W, BGR order) with mean and standard deviation
    
    for each channel in torch tensor:
        ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean: sequence of means for each channel
        std: sequence of stds for each channel
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace
    
    def __call__(self, img):
        """
        Args:
            (H W C) format numpy array range from [0, 255]
        Returns:
            (H W C) format numpy array in float32 range from [0, 1]
        """        
        assert torch.is_tensor(img) and img.ndimension() == 3, 'not an image tensor'

        if not self.inplace:
            img = img.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)
        std = torch.tensor(self.std, dtype=torch.float32)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])

        return img

class CenterCrop:
    """resize each image’s shorter edge to r pixels while keeping its aspect ratio. 
    Next, we crop out the cropped region in the center 
    Args:
        resized: resize image' shorter edge to resized pixels while keeping the aspect ratio
        cropped: output image size(h, w), if cropped is an int, then output cropped * cropped size
                 image
    """

    def __init__(self, cropped, resized=256, interpolation='linear'):

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

class RandomErasing:
    """Random erasing the an rectangle region in Image.
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.

    Args:
        sl: min erasing area region 
        sh: max erasing area region
        r1: min aspect ratio range of earsing region
        p: probability of performing random erasing
    """

    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3):

        self.p = p
        self.s = (sl, sh)
        self.r = (r1, 1/r1)
    

    def __call__(self, img):
        """
        perform random erasing
        Args:
            img: opencv numpy array in form of [w, h, c] range 
                 from [0, 255]
        
        Returns:
            erased img
        """

        assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'

        if random.random() > self.p:
            return img
        
        else:
            while True:
                Se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
                re = random.uniform(*self.r) 

                He = int(round(math.sqrt(Se * re)))
                We = int(round(math.sqrt(Se / re)))

                xe = random.randint(0, img.shape[1])
                ye = random.randint(0, img.shape[0])

                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    img[ye : ye + He, xe : xe + We, :] = np.random.randint(low=0, high=255, size=(He, We, img.shape[2]))

                    return img

class CutOut:
    """Randomly mask out one or more patches from an image. An image
    is a opencv format image (h,w,c numpy array)

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length, n_holes=1):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):

        while self.n_holes:

            y = random.randint(0, img.shape[0] - 1)
            x = random.randint(0, img.shape[1] - 1)

            tl_x = int(max(0, x - self.length / 2))
            tl_y = int(max(0, y - self.length / 2))

            img[tl_y : tl_y + self.length, tl_x : tl_x + self.length, :] = 0

            self.n_holes -= 1
        
        return img


