import itertools
import os.path as osp
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .base_derain_dataset import BaseDerainDataset
from .registry import DATASETS


@DATASETS.register_module()
class DerainFilenameMatchingDataset(BaseDerainDataset):
    """General paired image folder dataset for image deraining.

    The dataset is used when rainy images and corresponding labels are
    in different folder. Rainy images and labels are paired by filename
    matching, including full matching and prefix matching.

    In the training stage, it assumes that the path to rainy images
    is '{dataroot}/train/{lq_folder}' and the path to background images
    is '{dataroot}/train/{gt_folder}' During test times, the path to rainy
    and background images are '{dataroot}/test/{lq_folder}',
    '{dataroot}/test/{gt_folder}' respectively.

    Example for filename full matching:

    ::
        ${data_root}
        ├── train
        │   ├──${lq_folder}
        │   │   ├──1.png
        │   │   ├──2.png
        │   ├──${gt_folder}
        │   │   ├──1.png
        │   │   ├──2.png
        ├── test
        │   ├──${lq_folder}
        │   │   ├──1.png
        │   │   ├──2.png
        │   ├──${gt_folder}
        │   │   ├──1.png
        │   │   ├──2.png

    Example for filename prefix matching

    ::
        ${data_root}
        ├── train
        │   ├──${lq_folder}
        │   │   ├──1_1.png
        │   │   ├──1_2.png
        │   │   ├──1_3.png
        │   ├──${gt_folder}
        │   │   ├──1.png
        ├── test
        │   ├──${lq_folder}
        │   │   ├──1_1.png
        │   │   ├──1_2.png
        │   │   ├──1_3.png
        │   ├──${gt_folder}
        │   │   ├──1.png

    Args:
        dataroot (str | :obj:`Path`): Path to the folder root of images.
        lq_folder (str | :obj:`Path`): Folder name to rainy images.
        gt_folder (str | :obj:`Path`): Folder name to background images.
        pipeline (list[dict | callable]): A sequence of data transformations.
        mapping_rule (str): Matching rule of filename, should be one of 'full'
            and 'prefix'. If `mapping_rule` is 'prefix', then `serarator` must
            be passed in. Default: 'full'
        separator (optional, str): Separator to split prefix from filename.
        test_mode (bool): Store `True` when building test dataset.
            Default: `False`.
    """

    def __init__(
        self,
        dataroot: Union[str, Path],
        lq_folder: str,
        gt_folder: str,
        pipeline: List[Union[Dict, Callable]],
        mapping_rule: str = 'full',
        separator: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        super().__init__(pipeline, test_mode)
        phase = 'test' if test_mode else 'train'
        self.dataroot_lq = osp.join(str(dataroot), phase, lq_folder)
        self.dataroot_gt = osp.join(str(dataroot), phase, gt_folder)
        self.data_infos_lq = self.load_annotations(self.dataroot_lq)
        self.data_infos_gt = self.load_annotations(self.dataroot_gt)

        valid_mapping_rule = {'full', 'prefix'}
        if mapping_rule not in valid_mapping_rule:
            raise ValueError(f'supported mapping rules are {valid_mapping_rule},\
                but got mapping_rule={mapping_rule}')

        if mapping_rule == 'prefix' and separator is None:
            raise ValueError('separator should be specified when using prefix matching')

        self.mapping_table = self.generate_mapping_table(
            mapping_rule,
            separator
        )

    def load_annotations(self, dataroot: str) -> List[Dict]:
        """Load image paths of one domain.

        Args:
            dataroot (str): Path to the folder root for images.

        Returns:
            list[dict]: List that contains image paths of one domain.
        """
        data_infos = []
        paths = sorted(self.scan_folder(dataroot))
        for path in paths:
            data_infos.append(dict(path=path))
        return data_infos

    def prepare_train_data(self, idx: int) -> List[Dict]:
        """Prepare paired training data.

        Args:
            idx (int): Index of current batch.

        Returns:
            dict: Prepared training data batch.
        """
        img_lq_path, img_gt_path = self.mapping_table[idx]
        results = dict(lq_path=img_lq_path, gt_path=img_gt_path)
        return results

    def prepare_test_data(self, idx: int) -> List[Dict]:
        """Prepare paired test data.

        Args:
            idx (int): Index of current batch.

        Returns:
            list[dict]: Prepared test data batch.
        """
        img_lq_path, img_gt_path = self.mapping_table[idx]
        results = dict(lq_path=img_lq_path, gt_path=img_gt_path)
        return results

    def generate_mapping_table(self, rule, separator) -> Dict:
        combinations = itertools.product(
            [elem['path'] for elem in self.data_infos_lq],
            [elem['path'] for elem in self.data_infos_gt]
        )

        def get_basename(path):
            return osp.splitext(osp.split(path)[-1])[0]

        def prefix_match(x):
            lq_path, gt_path = x
            basename_lq = get_basename(lq_path)
            basename_gt = get_basename(gt_path)

            prefix = basename_lq.split(separator)[0]
            return basename_gt == prefix

        def full_match(x):
            lq_path, gt_path = x
            basename_lq = get_basename(lq_path)
            basename_gt = get_basename(gt_path)

            return basename_lq == basename_gt

        matching = None
        if rule == 'prefix':
            matching = prefix_match
        elif rule == 'full':
            matching = full_match
        else:
            raise ValueError(f'Illegal argument rule={rule}')  # Should not be reached

        result = list(filter(matching, combinations))
        return result

    def __len__(self) -> int:
        return len(self.mapping_table)
