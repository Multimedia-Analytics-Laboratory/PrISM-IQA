import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from typing import Optional, Sequence


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        global I
        I = Image.open(image_name)
    return I.convert('RGB')


def get_default_img_loader():
    return functools.partial(image_loader)


class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 num_problems=6,
                 label_columns: Optional[Sequence[str]] = None,
                 ignored_label_columns: Optional[Sequence[str]] = None,
                 get_loader=get_default_img_loader):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory of the images.
            transform (callable, optional): transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file, sep=',')
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test
        self.num_problems = num_problems
        self.label_columns = list(label_columns) if label_columns is not None else None
        self.ignored_label_columns = set(ignored_label_columns or [])
        self.image_column = 'image' if 'image' in self.data.columns else self.data.columns[0]
        self.mapping = {
            0: [1, 0, 0, 0],
            1: [0, 1, 0, 0],
            2: [0, 0, 1, 0],
            3: [0, 0, 0, 1]
        }

    def _label_to_level_and_observed(self, value, column: Optional[str] = None):
        """
        The merged CSV uses -1 for N/A local-region questions.
        The returned level is folded to 0 for tensor shape consistency, while
        observed=False lets the CRF objective and metrics marginalize it out.
        """
        if column in self.ignored_label_columns:
            return 0, False
        if pd.isna(value):
            return 0, False

        label = int(value)
        if label < 0:
            return 0, False
        if label > 3:
            raise ValueError(f"unexpected ordinal label {label}, expected values in [-1, 3]")
        return label, True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            samples: a Tensor that represents a video segment.
        """
        image_name = os.path.join(self.img_dir, self.data.loc[index, self.image_column])
        img = self.loader(image_name)
        img = self.preprocess(img)
        img = img.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (img.size(2) >= 1024) | (img.size(3) >= 1024):
            step = 48
        else:
            step = 32
        patches = (img.unfold(2, kernel_h, step).unfold(3, kernel_w, step).
                   permute(2, 3, 0, 1, 4, 5).reshape(-1, n_channels, kernel_h, kernel_w))

        assert patches.size(0) >= self.num_patch
        #self.num_patch = np.minimum(patches.size(0), self.num_patch)
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]

        attributes = []
        observed = []
        if self.data.shape[1] > 1:
            if self.label_columns is not None:
                for column in self.label_columns:
                    level, is_observed = self._label_to_level_and_observed(self.data.loc[index, column], column)
                    attributes.append(level)
                    observed.append(is_observed)
            else:
                for i in range(self.num_problems):
                    column = self.data.columns[i + 1]
                    level, is_observed = self._label_to_level_and_observed(self.data.iloc[index, i + 1], column)
                    attributes.append(level)
                    observed.append(is_observed)
            mapped_values = [self.mapping[attrib] for attrib in attributes]
            # all_node = np.array(mapped_values).reshape(-1)
            all_node = np.array(mapped_values)
        else:
            all_node = np.array([])
            observed = []

        sample = {
            'I': patches,
            'filename': self.data.loc[index, self.image_column],
            'all_node': all_node,
            'observed_mask': np.array(observed, dtype=np.bool_)
        }

        return sample

    def __len__(self):
        return len(self.data.index)
