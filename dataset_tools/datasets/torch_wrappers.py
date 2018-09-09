""" Wrapper to transform datasets into torch.utils.data.Dataset """

import torch
import torch.utils.data
import torchvision

from .detection_dataset import DetectionDataset
from .. import transforms


class ImageCollateContainer(object):
    """ Images in group of entries can be of different sizes
    We need a collate function to pad the images to make them all the same size
    This is a container for collate function that stores all the collate variables
    Used to access the collate_fn for use in dataset_loader creation

    Args
        pad_method (str, optional): For images of different sizes, states the
            way to pad images to create a batch of same size.
            Option of ['top-left', 'center']. Default 'top-left'.
        mode (str, optional): 'constant', 'reflect' or 'replicate'. Default: 'constant'
        value (int, optional): fill value for constant padding
    """
    def __init__(self, pad_method='top-left', mode='constant', value=0):
        assert pad_method in ['top-left', 'center'], 'pad_method must be either "top-left" or "center"'
        self.pad_method = pad_method
        self.mode = mode
        self.value = value

    def collate_fn(self, entries):
        all_shapes = [e['image'].shape for e in entries]
        max_shape = [max(s[i] for s in all_shapes) for i in range(3)]

        for entry, entry_shape in zip(entries, all_shapes):
            if self.pad_method == 'center':
                x1_pad = int((max_shape[2] - entry_shape[2]) / 2)
                y1_pad = int((max_shape[1] - entry_shape[1]) / 2)
                x2_pad = max_shape[2] - entry_shape[2] - x1_pad
                y2_pad = max_shape[1] - entry_shape[1] - y1_pad
            else:
                x1_pad = 0
                y1_pad = 0
                x2_pad = max_shape[2] - entry_shape[2]
                y2_pad = max_shape[1] - entry_shape[1]

            entry['image'] = torch.nn.functional.pad(
                entry['image'],
                pad=(x1_pad, y1_pad, x2_pad, y2_pad),
                mode=self.mode,
                value=self.value
            )

            if 'mask' in entry:
                entry['mask'] = torch.nn.functional.pad(
                    entry['mask'],
                    pad=(x1_pad, y1_pad, x2_pad, y2_pad),
                    mode=self.mode,
                    value=self.value
                )

            if 'annotations' in entry:
                if len(entry['annotations']) > 0:
                    entry['annotations'][:, 0] += y1_pad
                    entry['annotations'][:, 1] += x1_pad
                    entry['annotations'][:, 2] += y2_pad
                    entry['annotations'][:, 3] += x2_pad

        batch = { 'image': torch.stack([e['image'] for e in entries], dim=0) }

        if 'mask' in entry:
            batch['mask'] = torch.stack([e['mask'] for e in entries], dim=0)

        if 'annotations' in entry:
            batch['annotations'] = [e['annotations'] for e in entries]

        return batch


class TorchDetectionDataset(torch.utils.data.Dataset):
    """ Torch wrapper for detection dataset

    Args
        dataset_file : JSON file containing the detection dataset
        transform    : transformation to apply to each sample
    """
    def __init__(self, dataset_file, transform=None):
        self.dataset = DetectionDataset(dataset_file)
        self.transform = transform

        self.all_idx = self.dataset.get_all_image_index()
        self.image_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.all_idx)

    def __getitem__(self, idx):
        image_id = self.all_idx[idx]
        sample = {
            'image_id'   : image_id,
            'image'      : self.dataset.get_image_pil(image_id),
            'annotations': self.dataset.get_ann_array(image_id)
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['image'] = self.image_to_tensor(sample['image'])
        sample['annotations'] = torch.from_numpy(sample['annotations'])

        return sample


class TorchSegmentationDataset(torch.utils.data.Dataset):
    """ Torch wrapper for segmentation dataset

    Args
        dataset_file : JSON file containing the detection dataset
        transform    : transformation to apply to each sample
    """
    def __init__(self, dataset_file, transform=None):
        self.dataset = DetectionDataset(dataset_file)
        self.transform = transform

        self.all_idx = self.dataset.get_all_ann_index()
        self.apply_bbox = transforms.ApplyBbox()
        self.image_to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.mask_to_tensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.all_idx)

    def __getitem__(self, idx):
        ann_id = self.all_idx[idx]
        sample = {
            'ann_id': ann_id,
            'image': self.dataset.get_image_pil(ann_id=ann_id),
            'bbox': self.dataset.get_ann_array(ann_id=ann_id),
            'mask': self.dataset.get_mask_pil(ann_id=ann_id),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        # Apply bbox if needed
        if 'bbox' in sample:
            sample = self.apply_bbox(sample)

        sample['image'] = self.image_to_tensor(sample['image'])
        sample['mask'] = self.mask_to_tensor(sample['mask'])

        return sample
