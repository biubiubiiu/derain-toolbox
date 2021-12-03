# This code is taken from https://github.com/open-mmlab/mmediting
# Modified by Raymond Wong

import os.path as osp
from pathlib import Path
from typing import Callable, Dict, List, Union

from .base_derain_dataset import BaseDerainDataset
from .registry import DATASETS


@DATASETS.register_module()
class DerainPairedDataset(BaseDerainDataset):
    """General paired image folder dataset for image deraining.

    It assumes that the training directory is '/path/to/data/train'.
    During test time, the directory is '/path/to/data/test'. '/path/to/data'
    can be initialized by args 'dataroot'. Each sample contains a pair of
    images concatenated in the w dimension (A|B)

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of paired images.
        pipeline (List[dict | callable]): A sequence of data transformations.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`
    """

    def __init__(
        self,
        dataroot: Union[str, Path],
        pipeline: List[Union[Dict, Callable]],
        test_mode: bool = False
    ) -> None:
        super().__init__(pipeline, test_mode)
        phase = 'test' if test_mode else 'train'
        self.dataroot = osp.join(str(dataroot), phase)
        self.data_infos = self.load_annotations()

    def load_annotations(self) -> List[Dict]:
        """Load paired image paths.

        Returns:
            list[dict]: List that contains paired image paths.
        """
        data_infos = []
        pair_paths = sorted(self.scan_folder(self.dataroot))
        for pair_path in pair_paths:
            data_infos.append(dict(pair_path=pair_path))

        return data_infos
