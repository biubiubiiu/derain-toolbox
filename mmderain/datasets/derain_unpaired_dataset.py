# This code is taken from https://github.com/open-mmlab/mmediting
# Modified by Raymond Wong

import os.path as osp
import random
from pathlib import Path
from typing import Callable, Dict, List, Union

from .base_derain_dataset import BaseDerainDataset
from .registry import DATASETS


@DATASETS.register_module()
class DerainUnpairedDataset(BaseDerainDataset):
    """General unpaired image folder dataset for image deraining.

    It assumes that the training directory is '/path/to/data/train'.
    During test time, the directory is '/path/to/data/test'. '/path/to/data'
    is initialized by args 'dataroot'. The directory of images from domain
    A, B is specified by args 'dataroot_a' and 'dataroot_b', respectively.
    (i.e. traning directory to each domain is '/path/to/data/train/{dataroot_a}'
    and '/path/to/data/train/{dataroot_b}', testing directory to each domain is
    '/path/to/data/test/{dataroot_a}' and '/path/to/data/test/{dataroot_b}'

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of unpaired
            images.
        dataroot_a (str): Directory name of domain A images
        dataroot_b (str): Directory name of domain B images
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(
        self,
        dataroot: Union[str, Path],
        dataroot_a: str,
        dataroot_b: str,
        pipeline: List[Union[Dict, Callable]],
        test_mode: bool = False
    ) -> None:
        super().__init__(pipeline, test_mode)
        phase = 'test' if test_mode else 'train'
        self.dataroot_a = osp.join(str(dataroot), phase, dataroot_a)
        self.dataroot_b = osp.join(str(dataroot), phase, dataroot_b)
        self.data_infos_a = self.load_annotations(self.dataroot_a)
        self.data_infos_b = self.load_annotations(self.dataroot_b)
        self.len_a = len(self.data_infos_a)
        self.len_b = len(self.data_infos_b)

    def load_annotations(self, dataroot: str) -> List[Dict]:
        """Load unpaired image paths of one domain.

        Args:
            dataroot (str): Path to the folder root for unpaired images of
                one domain.

        Returns:
            list[dict]: List that contains unpaired image paths of one domain.
        """
        data_infos = []
        paths = sorted(self.scan_folder(dataroot))
        for path in paths:
            data_infos.append(dict(path=path))
        return data_infos

    def prepare_train_data(self, idx: int) -> Dict:
        """Prepare unpaired training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        idx_b = random.randrange(0, self.len_b)
        img_b_path = self.data_infos_b[idx_b]['path']
        results = dict(img_a_path=img_a_path, img_b_path=img_b_path)
        return results

    def prepare_test_data(self, idx: int) -> List[Dict]:
        """Prepare unpaired test data.

        Args:
            idx (int): Index of current batch.

        Returns:
            list[dict]: Prepared test data batch.
        """
        img_a_path = self.data_infos_a[idx % self.len_a]['path']
        img_b_path = self.data_infos_b[idx % self.len_b]['path']
        results = dict(img_a_path=img_a_path, img_b_path=img_b_path)
        return results

    def __len__(self):
        return max(self.len_a, self.len_b)
