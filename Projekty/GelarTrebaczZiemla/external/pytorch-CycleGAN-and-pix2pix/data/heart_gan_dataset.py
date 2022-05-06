import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms 
import numpy as np
import nibabel as nib
import glob
import os

from data import BaseDataset


class HeartGanDataset(BaseDataset):
    def __init__(self, opt):
        """
        Creates dataset with augumentation from nifti files in a directory
        Args:
            images_directory_ct: path of directory containing images
            kind: one of ("ct", "mr") - type of images in dataset
            rotation_degrees: Range of degrees to select from.
                If degrees is a number instead of sequence like (min, max), the range of degrees
                will be (-degrees, +degrees).
            size: desired size; if None no resizing is done
            h_flip: if True horizontal flip is randomly applied 
            v_flip: if True vertical flip is randomly applied
        """
        images_directory_ct = "datasets/heart/affregcommon2mm_roi_ct_train"
        images_directory_mr = "datasets/heart/affregcommon2mm_roi_mr_train"
        rotation_degrees = 5
        size = (opt.load_size, opt.load_size)
        h_flip = True
        v_flip = True
        #python train.py --dataroot ./datasets/heart --name heart_cyclegan --model cycle_gan --dataset_mode=heart_gan --lambda_A=10 --lambda_B=10 --lambda_identity=0 --batch_size=10 --beta1=0.9 --input_nc=1 --output_nc=1 --num_threads=12

        self.LABEL_ENCODING_ARRAY = np.array([  0, 205, 500, 600, 420, 550, 820, 850]).reshape((-1, 1, 1))
        self.SLICES_PER_SCAN = 96
        self.images_paths_ct = glob.glob(images_directory_ct + os.sep + "*image*")
        self.images_paths_mr = glob.glob(images_directory_mr + os.sep + "*image*")
        # self.labels_paths_ct = [name.replace("_image.nii.gz", "_label.nii.gz") for name in self.images_paths_ct]
        # self.labels_paths_mr = [name.replace("_image.nii.gz", "_label.nii.gz") for name in self.labels_paths_mr]
        random_transforms = [transforms.RandomRotation(rotation_degrees)]
        if h_flip:
            random_transforms.append(transforms.RandomHorizontalFlip())
        if v_flip:
            random_transforms.append(transforms.RandomVerticalFlip())
        self.random_transform = transforms.Compose(random_transforms) 
        self.resize = size
        if size is not None:
            # self.label_resizer = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
            self.image_resizer = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
        self.normalize_ct = transforms.Normalize(-221.20035285101997, 439.02997204491294)
        self.normalize_mr = transforms.Normalize(261.89050726996527, 321.0735487258476)

    def __len__(self):
        return len(self.images_paths_ct) * self.SLICES_PER_SCAN

    def __getitem__(self, idx):
        scan_ct = nib.load(self.images_paths_ct[idx // self.SLICES_PER_SCAN])
        # label_scan_ct = nib.load(self.labels_paths_ct[idx // self.SLICES_PER_SCAN])
        index_ct = idx % self.SLICES_PER_SCAN
        scan_slice_ct, label_slice_ct = self.transform(scan_ct, None, index_ct, is_ct=True)
        idx_mr = random.randint(0, len(self.images_paths_mr) * self.SLICES_PER_SCAN - 1)
        scan_mr = nib.load(self.images_paths_mr[idx_mr // self.SLICES_PER_SCAN])
        # label_scan_mr = nib.load(self.labels_paths_ct[idx_mr // self.SLICES_PER_SCAN])
        index_mr = idx_mr % self.SLICES_PER_SCAN
        scan_slice_mr, label_slice_mr = self.transform(scan_mr, None, index_mr, is_ct=False)
        A_path = self.images_paths_ct[idx // self.SLICES_PER_SCAN]
        B_path = self.images_paths_mr[idx // self.SLICES_PER_SCAN]
        return {"A": scan_slice_ct, "B": scan_slice_mr, 'A_paths': A_path, 'B_paths': B_path}#"A_label": label_slice_ct, "B_label": label_slice_mr}

    def transform(self, scan, label_scan, index, is_ct = True):
        #reading data
        scan_slice = scan.slicer[:,:, index:index+1].get_fdata()
        # label_slice = label_scan.slicer[:,:, index:index+1].get_fdata()
        # label_slice = torch.from_numpy(label_slice).reshape(1, 96, 80)
        scan_slice = torch.from_numpy(scan_slice).reshape(1, 96, 80).float()
        scan_slice = self.normalize_ct(scan_slice) if is_ct else self.normalize_mr(scan_slice)
        # random transformations (images and labels are concatenated to ensure the same transformations are applied to them)
        # cated_tensor = torch.cat((scan_slice, label_slice), 0)
        # cated_tensor = self.random_transform(cated_tensor)
        scan_slice - self.random_transform(scan_slice)
        # scan_slice, label_slice= cated_tensor[[0],:,:], cated_tensor[[1],:,:]
        # label_slice_encoded = np.equal(label_slice, self.LABEL_ENCODING_ARRAY)      #onehot encoding label
        if self.resize is not None:
            scan_slice = self.image_resizer(scan_slice)
            # label_slice_encoded = self.label_resizer(label_slice_encoded)
        return scan_slice, None#label_slice_encoded


