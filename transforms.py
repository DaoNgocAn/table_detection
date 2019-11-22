import random
import torch
import ocrodeg
import scipy.ndimage as ndi

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class OneOf(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        t = random.choice(self.transforms)
        image, target = t(image, target)
        return image, target



class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class RandomDistortions(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            sigma = random.random() * 20
            noise = ocrodeg.bounded_gaussian_noise(image.shape[:2], sigma, 5.0)
            distorted = ocrodeg.distort_with_noise(image, noise)
            image = distorted
        return image, target


class Blur(object):
    def __init__(self, prob, kernel_size_max=2):
        self.prob = prob
        self.kernel_size_max = kernel_size_max

    def __call__(self, image, target):
        if random.random() < self.prob:
            kernel_size = random.randint(0, self.kernel_size_max)
            image = ndi.gaussian_filter(image, kernel_size)
        return image, target


class BlurThreshold(object):
    def __init__(self, prob, kernel_size_max=2, threshold=0.5):
        self.prob = prob
        self.kernel_size_max = kernel_size_max
        self.threshold = threshold

    def __call__(self, image, target):
        if random.random() < self.prob:
            kernel_size = random.randint(0, self.kernel_size_max)
            image = ndi.gaussian_filter(image, kernel_size)
            image = 1.0 * (image>self.threshold)

        return image, target

class BinaryBlur(object):
    def __init__(self, prob, kernel_size_max=2):
        self.prob = prob
        self.kernel_size_max =kernel_size_max

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = ocrodeg.binary_blur(image, random.randint(0, self.kernel_size_max))
        return image, target



class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
