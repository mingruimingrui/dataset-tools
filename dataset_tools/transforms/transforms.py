""" Copy of https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py

But for dict entries of the following format

{
    'image': PIL.Image object,
    'bbox' (optional): (x1, y1, x2, y2) coords,
    'annotations' (optional): np.ndarray in shape (N, 5),
    'mask' (optional): PIL.Image object,
    ...
}

Other keys will simply be converted into tensors
"""
from __future__ import division

import random
import numbers
import collections
from copy import deepcopy

import numpy as np
import cv2
from PIL import Image

import torchvision.transforms.functional as F


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}


_pil_interpolation_to_cv2 = {
    Image.NEAREST: cv2.INTER_NEAREST,
    Image.BILINEAR: cv2.INTER_LINEAR,
    Image.BICUBIC: cv2.INTER_CUBIC,
    Image.LANCZOS: cv2.INTER_LANCZOS4,
}


class ApplyBbox(object):
    """ Applies the bbox for each entry on the input sample

    Args:
        expand (float, optional): Percent to expand bounding box by in ratio form
            eg. expand=0.2 means bounding box will have it's height and width each increased by 20%
        filter_annotations (bool, optional): Flag to raise if annotations should
            be filtered post cropping
    """
    def __init__(self, expand=0, filter_annotations=True):
        self.expand = expand
        self.filter_annotations = filter_annotations

    def __call__(self, entry):
        # Make copy of entry
        entry = deepcopy(entry)

        if 'bbox' not in entry:
            return entry

        # Extract bbox
        bbox = entry['bbox']
        del entry['bbox']

        # Expand bbox
        if self.expand != 0:
            bbox_w = bbox[2] - bbox[0]
            bbox_h = bbox[3] - bbox[1]
            expand_w = bbox_w * self.expand / 2
            expand_h = bbox_h * self.expand / 2
            bbox[0] -= expand_w
            bbox[1] -= expand_h
            bbox[2] += expand_w
            bbox[3] += expand_h

        # apply bbox to image and mask
        entry['image'] = entry['image'].crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        if 'mask' in entry:
            entry['mask'] = entry['mask'].crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        if 'annotations' in entry:
            # apply bbox to annotations
            anns = entry['annotations'].astype('float')
            anns[:, 0] -= bbox[1]
            anns[:, 1] -= bbox[0]
            anns[:, 2] -= bbox[1]
            anns[:, 3] -= bbox[0]

            if self.filter_annotations:
                new_image_size = entry['image'].size
                keep_idx = (anns[:, 0] >= 0) & \
                           (anns[:, 1] >= 0) & \
                           (anns[:, 2] <= new_image_size[0]) & \
                           (anns[:, 3] <= new_image_size[1])
                anns = anns[keep_idx]

            entry['annotations'] = anns

        return entry

    def __repr__(self):
        return self.__class__.__name__ + '(expand={}, filter_annotations={})'.format(self.expand, self.filter_annotations)


class ImageCompose(object):
    """ Composes several transforms together and apply only to the Image """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, entry):
        for t in self.transforms:
            entry['image'] = t(entry['image'])
        return entry

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize(object):
    """ Resize input sample to given size

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, entry):
        # Make copy of entry
        entry = deepcopy(entry)
        orig_image_size = entry['image'].size

        # Rescale image
        entry['image'] = F.resize(entry['image'], self.size, self.interpolation)
        new_image_size = entry['image'].size

        # Get height and width scale
        h_scale = new_image_size[1] / orig_image_size[1]
        w_scale = new_image_size[0] / orig_image_size[0]

        # Resize rest of fields
        for k, v in entry.items():
            if k == 'mask':
                entry[k] = F.resize(v, self.size, Image.NEAREST)
            elif k == 'annotations':
                v = v.astype('float')
                v[:, 0] *= w_scale
                v[:, 1] *= h_scale
                v[:, 2] *= w_scale
                v[:, 3] *= h_scale
                entry[k] = v
            elif k == 'bbox':
                v = np.array(v, dtype='float')
                v[0] *= w_scale
                v[1] *= h_scale
                v[2] *= w_scale
                v[3] *= h_scale
                entry[k] = v

        return entry

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Pad(object):
    """Pad the given PIL Image on all sides with the given "pad" value.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.
            - constant: pads with a constant value, this value is specified with fill
            - edge: pads with the last value at the edge of the image
            - reflect: pads with reflection of image without repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
            - symmetric: pads with reflection of image repeating the last value on the edge
                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, entry):
        # Make copy of entry
        entry = deepcopy(entry)

        # Apply padding
        for k, v in entry.items():
            if k == 'image':
                entry[k] = F.pad(v, self.padding, self.fill, self.padding_mode)
            elif k == 'mask':
                entry[k] = F.pad(v, self.padding, 0, self.padding_mode)
            elif k == 'annotations':
                v[:, 0] += self.padding[0]
                v[:, 1] += self.padding[1]
                v[:, 2] += self.padding[0]
                v[:, 3] += self.padding[1]
                entry[k] = v
            elif k == 'bbox':
                v[0] += self.padding[0]
                v[1] += self.padding[1]
                v[2] += self.padding[0]
                v[3] += self.padding[1]
                entry[k] = v

        return entry

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, entry):
        # Make copy of entry
        entry = deepcopy(entry)
        orig_image_size = entry['image'].size

        if random.random() < self.p:
            for k, v in entry.items():
                if k in ['image', 'mask']:
                    entry[k] = F.hflip(v)
                elif k == 'annotations':
                    v[:, 0] = orig_image_size[0] - v[:, 0]
                    v[:, 2] = orig_image_size[0] - v[:, 2]
                    entry[k] = v
                elif k == 'bbox':
                    v[0] = orig_image_size[0] - v[0]
                    v[2] = orig_image_size[0] - v[2]
                    entry[k] = v

        return entry

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, entry):
        # Make copy of entry
        entry = deepcopy(entry)
        orig_image_size = entry['image'].size

        if random.random() < self.p:
            for k, v in entry.items():
                if k in ['image', 'mask']:
                    entry[k] = F.vflip(v)
                elif k == 'annotations':
                    v[:, 1] = orig_image_size[1] - v[:, 1]
                    v[:, 3] = orig_image_size[1] - v[:, 3]
                    entry[k] = v
                elif k == 'bbox':
                    v[1] = orig_image_size[1] - v[1]
                    v[3] = orig_image_size[1] - v[3]
                    entry[k] = v

        return entry

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


# class FixedRandomRotation(object):
#     """Rotates the given PIL Image at a fixed angle with a given probability
#     Args:
#         angle (float, int): angle in degree to rorate the image in the
#             clockwise direction. Default value is 90
#         p (float): probability of the image being rotated. Default value is 0.5
#         resample (int, optional): Desired interpolation. Default is
#             ``PIL.Image.BILINEAR``
#         expand (bool, optional): Optional expansion flag.
#             If true, expands the output image to make it large enough to hold the entire rotated image.
#             If false or omitted, make the output image the same size as the input image.
#             Note that the expand flag assumes rotation around the center and no translation.
#     """
#     def __init__(self, angle=90, p=0.5, resample=Image.BILINEAR, expand=False):
#         self.angle = angle
#         self.p = p
#         self.resample = self.resample
#         self.expand = expand
#
#     def __call__(self, entry):
#         # Make copy of entry
#         entry = deepcopy(entry)
#
#         if random.random() < self.p:
#             for k, v in entry.items():
#                 if k in ['image', 'mask']:
#                     entry[k] = F.rotate(v, self.angle, resample=self.resample, expand=self.expand)
#                 elif k == 'annotations':
#
#                 elif k == 'bbox':
#
#
#         return entry
#
#     def __repr__(self):
#         interpolate_str = _pil_interpolation_to_str[self.resample]
#         return self.__class__.__name__ + '(angle={}, p={}, resample={}, expand={})'.format(
#             self.angle, self.p, interpolate_str, self.expand)
