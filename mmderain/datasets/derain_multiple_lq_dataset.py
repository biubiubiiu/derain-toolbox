import itertools
import os.path as osp
from pathlib import Path
from typing import Callable, Dict, List, Union

from .base_derain_dataset import BaseDerainDataset
from .registry import DATASETS


@DATASETS.register_module()
class DerainMultipleLQDataset(BaseDerainDataset):
    """General paired image folder dataset for image deraining.

    The dataset is used when multiple LQ images corresponds to one
    GT image (e.g. Rain1400). It assumes that the training directory
    is '/path/to/data/train'. During test time, the directory is
    '/path/to/data/test'.

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of unpaired
            images.
        lq_folder (str | :obj:`Path`): Path to a lq folder.
        gt_folder (str | :obj:`Path`): Path to a gt folder.
        num_input_frames (int): Number of input frames.
        pipeline (list[dict | callable]): A sequence of data transformations.
        scale (int): Upsampling scale ratio.
        val_partition (str): Validation partition mode. Choices ['official' or
        'REDS4']. Default: 'official'.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(
        self,
        dataroot: Union[str, Path],
        lq_folder: str,
        gt_folder: str,
        pipeline: List[Union[Dict, Callable]],
        mapping_rule: str = 'prefix',
        test_mode: bool = False,
        **kwargs
    ) -> None:
        super().__init__(pipeline, test_mode)
        phase = 'test' if test_mode else 'train'
        self.dataroot_lq = osp.join(str(dataroot), phase, lq_folder)
        self.dataroot_gt = osp.join(str(dataroot), phase, gt_folder)
        self.data_infos_lq = self.load_annotations(self.dataroot_lq)
        self.data_infos_gt = self.load_annotations(self.dataroot_gt)

        valid_mapping_rule = {'prefix'}
        if mapping_rule not in valid_mapping_rule:
            raise ValueError(f'supported mapping rules are {valid_mapping_rule},\
                but got mapping_rule={mapping_rule}')

        self.mapping_table = self.generate_mapping_table(
            mapping_rule,
            **kwargs
        )

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

    def prepare_train_data(self, idx):
        """Prepare unpaired training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        img_lq_path, img_gt_path = self.mapping_table[idx]
        results = dict(lq_path=img_lq_path, gt_path=img_gt_path)
        return self.pipeline(results)

    def prepare_test_data(self, idx: int) -> List[Dict]:
        """Prepare unpaired test data.

        Args:
            idx (int): Index of current batch.

        Returns:
            list[dict]: Prepared test data batch.
        """
        img_lq_path, img_gt_path = self.mapping_table[idx]
        results = dict(lq_path=img_lq_path, gt_path=img_gt_path)
        return self.pipeline(results)

    def generate_mapping_table(self, rule, **kwargs) -> Dict:
        if rule == 'prefix':
            separator = kwargs['separator']
            combinations = itertools.product(
                [elem['path'] for elem in self.data_infos_lq],
                [elem['path'] for elem in self.data_infos_gt]
            )

            def prefix_match(x):
                lq_path, gt_path = x
                filename_lq = osp.split(lq_path)[-1]
                basename_lq = osp.splitext(filename_lq)[0]
                filename_gt = osp.split(gt_path)[-1]
                basename_gt = osp.splitext(filename_gt)[0]

                prefix = basename_lq.split(separator)[0]
                return basename_gt == prefix

            result = list(filter(prefix_match, combinations))
            return result

        # Should not be reached
        raise ValueError(f'Illegal argument rule={rule}')

    def __len__(self):
        return len(self.mapping_table)
