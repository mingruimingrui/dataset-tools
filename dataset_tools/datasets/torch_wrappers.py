""" Wrapper to transform datasets into torch.utils.data.Dataset """

import torch
import torch.utils.data
import torchvision

from .detection_dataset import DetectionDataset


class TorchDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, transform=None):
        """ Torch wrapper for detection dataset

        Args
            dataset_file : JSON file containing the detection dataset
            transform    : transformation to apply to each sample
        """
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
        sample {
            'image_id'   : image_id,
            'image'      : self.dataset.get_image_pil(image_id),
            'annotations': self.get_ann_array(image_id)
        }

        if self.transform is not None:
            sample = self.transform(sample)

        sample['image'] = self.image_to_tensor(sample['image'])
        sample['annotations'] = torch.from_numpy(sample['annotations'])

        return sample


class TorchSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_file, transform=None):
        """ Torch wrapper for segmentation dataset

        Args
            dataset_file : JSON file containing the detection dataset
            transform    : transformation to apply to each sample
        """
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
