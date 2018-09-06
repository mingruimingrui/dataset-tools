""" File to store all dataset getter functions """

import os
import numpy as np
from copy import deepcopy

from ..utils.image_io import read_image, read_image_url


def get_dataset_file(self):
    """ Retrieves the dataset save path """
    return self.dataset_file


def get_size(self):
    """ Retrieves the number of images in dataset """
    return len(self.image_infos)


def get_root_dir(self):
    """ Retrieves the root directory to all images """
    return len(self.root_dir)


def get_num_classes(self):
    """ Retrieves the number of classes in dataset """
    return len(self.id_to_class_info)


def name_to_label(self, class_name):
    """ Retrieves the class id given a class name """
    return self.name_to_class_info[class_name]['id']


def label_to_name(self, class_id):
    """ Retrieves the class name given a class id """
    return self.id_to_class_info[class_id]['name']


def get_all_classes(self):
    """ Retrieves the list of classe names in this dataset """
    return [name for name in self.name_to_class_info.keys()]


def get_classes_dict(self):
    """ Retrieves the classes in dictionary form """
    return deepcopy(self.id_to_class_info)


def get_all_image_index(self):
    """ Retrieves all image ids """
    return list(self.image_infos.keys())


def get_all_ann_index(self):
    """ Retrieves all annotation ids """
    return list(self.ann_infos.keys())


def get_image_info(self, image_id=None, ann_id=None):
    """ Retrieves the full image info given an image_id or ann_id """
    if image_id is None:
        assert ann_id is not None, 'Either image_id or ann_id has to be provided'
        image_id = self.ann_infos[ann_id]['image_id']

    return deepcopy(self.image_infos[image_id])


def get_image_pil(self, image_id=None, ann_id=None):
    """ Retrieves the image associated with the image_id or ann_id """
    if image_id is None:
        assert ann_id is not None, 'Either image_id or ann_id has to be provided'
        image_id = self.ann_infos[ann_id]['image_id']

    if self.image_infos[image_id]['image_path'] is None:
        image_url = self.image_infos[image_id]['image_url']
        return read_image_url(image_url)
    else:
        image_path = self.image_infos[image_id]['image_path']
        image_path = os.path.join(self.root_dir, image_path)
        return read_image(image_path)


def get_ann_info(self, image_id=None, ann_id=None):
    """ Retrieves the annotation informations given an image id or annotation id

    if image_id provided, returns a list of annotation infos
    if ann_id provided, returns a single annotation info
    """
    if image_id is not None:
        return deepcopy([self.ann_infos[ann_id] for ann_id in self.img_to_ann[image_id]])
    elif ann_id is not None:
        return deepcopy(self.ann_infos[ann_id])
    else:
        raise Exception('Either image_id or ann_id has to be provided')


def get_ann_array(self, image_id=None, ann_id=None):
    """ Retrieves the annotation bbox and category_id as an array given an image id or annotation id

    if image_id provided, returns a (num_annotations, 5) array of annotation bbox + class_id
    if ann_id provided, returns a (5, ) array of annotation bbox + class_id
    """
    if image_id is not None:
        anns = self.get_ann_info(image_id)
        if len(anns) > 0:
            return np.array([ann['bbox'] + [ann['class_id']] for ann in anns])
        else:
            return np.zeros((0, 5))
    elif ann_id is not None:
        ann = self.ann_infos[ann_id]
        return np.array(ann['bbox'] + [ann['class_id']])
    else:
        raise Exception('Either image_id or ann_id has to be provided')
