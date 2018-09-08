import os
import json
from collections import OrderedDict

from . import _getters


class DetectionDataset(object):
    def __init__(self, dataset_file, root_dir=None, classes=None, new_dataset=False):
        """
        Dataset for detection and segmentation tasks

        Either loads an existing dataset or prepares to create a new dataset
        - If loading existing dataset, root_dir and classes will be ignored
        - If creating new dataset, root_dir and classes must have values

        Args
            dataset_file : Path to detection dataset json file
            root_dir     : Path to root directory of all image files in dataset (Only used when creating new dataset)
            classes      : list of strings representing all classes
            new_dataset  : Flag to raise if creating new dataset (Automatically set to true if dataset_file does not exist)

        """
        self.dataset_file = dataset_file

        # Create data structures to store data
        self.id_to_class_info   = OrderedDict()
        self.name_to_class_info = OrderedDict()
        self.image_infos        = OrderedDict()
        self.ann_infos          = OrderedDict()
        self.img_to_ann         = OrderedDict()
        self.next_image_id      = 0

        if not os.path.isfile(self.dataset_file):
            new_dataset = True

        if new_dataset:
            assert root_dir is not None and classes is not None, 'If creating new dataset, both root_dir and classes must be provided'
            self._init_new_dataset(root_dir, classes)
            print('New dataset initialized')
        else:
            print('Loading dataset')
            self._load_dataset()
            print('Dataset loaded')

    def _init_new_dataset(self, root_dir, classes):
        assert os.path.isdir(root_dir), '{} is not a valid path for root_dir'.format(root_dir)
        self.root_dir = root_dir

        for class_id, class_name in enumerate(classes):
            class_info = {
                'id'   : class_id,
                'name' : class_name
            }
            self.id_to_class_info[class_id]     = class_info
            self.name_to_class_info[class_name] = class_info

    def _load_dataset(self):
        with open(self.dataset_file, 'r') as f:
            data = json.load(f)

        # save root dir
        self.root_dir = data['root_dir']

        # retrieve class information
        for class_id, class_name in enumerate(data['classes']):
            class_info = {
                'id'  : class_id,
                'name': class_name
            }
            self.id_to_class_info[class_id]     = class_info
            self.name_to_class_info[class_name] = class_info

        # Retrieve image information
        for image_info in data['images']:
            self.image_infos[image_info['id']] = image_info
        self.next_image_id = max(self.image_infos.keys()) + 1

        # Config annotation infos such that it is retrievable through annotation id
        for ann_info in data['annotations']:
            self.ann_infos[ann_info['id']] = ann_info

        # Make the img_to_ann dict
        for image_info in data['images']:
            self.img_to_ann[image_info['id']] = []
        for ann_info in data['annotations']:
            self.img_to_ann[ann_info['image_id']].append(ann_info['id'])

    def save_dataset(self, dataset_file=None, force_overwrite=False):
        """ Save DetectionDataset to a json file
        Args
            dataset_file    : Path to DetectionDataset json file (or None if saving to same file dataset is loaded from)
            force_overwrite : Flag to raise if overwriting over existing dataset file
        """
        if dataset_file is not None:
            self.dataset_file = dataset_file

        assert self.dataset_file is not None

        # Initialize dict
        json_dataset = OrderedDict()

        # Save dataset info
        json_dataset['root_dir'] = self.root_dir
        json_dataset['classes'] = list(self.name_to_class_info.keys())
        json_dataset['images'] = list(self.image_infos.values())
        json_dataset['annotations'] = list(self.ann_infos.values())

        # Save dataset into json file
        if (not os.path.isfile(self.dataset_file)) or force_overwrite:
            print('Saving dataset as an annotation file, this can take a while')
            with open(self.dataset_file, 'w') as f:
                json.dump(json_dataset, f)
            print('Dataset saved')
        else:
            raise FileExistsError('Dataset not saved as it already exists, consider overwriting')

    ###########################################################################
    #### Dataset misc functions

    ###########################################################################
    #### Dataset getter and loaders

    get_dataset_file = _getters.get_dataset_file
    get_size         = _getters.get_size
    get_root_dir     = _getters.get_root_dir
    get_num_classes  = _getters.get_num_classes

    name_to_label = _getters.name_to_label
    label_to_name = _getters.label_to_name

    get_all_classes     = _getters.get_all_classes
    get_classes_dict    = _getters.get_classes_dict
    get_all_image_index = _getters.get_all_image_index
    get_all_ann_index   = _getters.get_all_ann_index

    get_image_info  = _getters.get_image_info
    get_image_pil   = _getters.get_image_pil
    get_image_array = _getters.get_image_array

    get_ann_info  = _getters.get_ann_info
    get_ann_array = _getters.get_ann_array

    get_mask_pil   = _getters.get_mask_pil
    get_mask_array = _getters.get_mask_array

    ###########################################################################
    #### Dataset setters

    def set_image(
        self,
        image_path=None,
        image_url=None,
        image_id=None,
        height=None,
        width=None,
        force_overwrite=False
    ):
        """ Sets an image entry in the dataset

        Required variables:
            image_path/image_url (atleast 1 required)

        Args
            image_path      : The path to the locally stored image relative to root_dir
            image_url       : The http public url to the image
            image_id        : An integer to use for the image id
            height          : The image pixel-wise height
            width           : The image pixel-wise width
            force_overwrite : Flag to trigger the overwrite of image at image_id
        Returns
            image info (Dataset object will also be updated with this new image info)
        """
        assert (image_url is not None) or (image_path is not None), 'Atleast one of image path or image url must be provided'

        # Identify image id
        if image_id is None:
            image_id = self.next_image_id
            self.next_image_id += 1
        else:
            assert isinstance(image_id, int), 'Image id if provided must be an integer, got {}'.format(type(image_id))
            assert (image_id not in self.image_infos) or force_overwrite, 'Image id {} already exists, consider overwrite'.format(image_id)
            self.next_image_id = max(self.next_image_id, image_id) + 1

        image_info = {
            'id'           : image_id,
            'image_path'   : image_path,
            'image_url'    : image_url,
            'width'        : width,
            'height'       : height
        }

        # Store all required info
        self.image_infos[image_id] = image_info
        self.img_to_ann[image_id] = []

        return image_info

    def set_ann(
        self,
        image_id,
        bbox,
        class_name=None,
        class_id=None,
        segmentation=None
    ):
        """ Sets a single image detection annotation, set_classes has to be ran in advanced

        Args
            image_id    : Image id associated to this detection
            bbox        : Bounding box for detection
            class_name  : Class name of object
            class_id    : Class id of object
            segmentation: RLE of the object mask
        """
        assert (class_name is not None) or (class_id is not None), 'Either class_name or class_id must be present'
        if class_name is not None:
            assert class_name in self.name_to_class_info
            class_id = self.name_to_label(class_name)
        else:
            assert class_id in self.id_to_class_info
            class_name = self.label_to_name(class_id)

        # Prepare ann_info
        ann_id = len(self.ann_infos)
        ann_info = {
            'id'          : ann_id,
            'image_id'    : image_id,
            'bbox'        : bbox,
            'class_id'    : class_id,
            'class_name'  : class_name,
            'segmentation': segmentation
        }

        # Store ann info
        self.ann_infos[ann_id] = ann_info
        self.img_to_ann[image_id] += [ann_id]

        return ann_info

    ###########################################################################
    #### Dataset editor
