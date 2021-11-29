# This code is taken from https://github.com/open-mmlab/mmediting
# Modified by Raymond Wong

from typing import Dict, List, Tuple, Union

import os.path as osp
from collections import defaultdict
from pathlib import Path

from mmcv import scandir

from .base_dataset import BaseDataset


IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')


class BaseDerainDataset(BaseDataset):
    """Base class for deraining dataset."""

    @staticmethod
    def scan_folder(path: Union[str, Path]) -> List[str]:
        """Obtain image path list (including sub-folders) from a given folder.

        Args:
            path (str | :obj:`Path`): Folder path.

        Returns:
            list[str]: Image list obtained from the given folder.
        """

        if isinstance(path, (str, Path)):
            path = str(path)
        else:
            raise TypeError("'path' must be a str or a Path object, "
                            f'but received {type(path)}.')

        images = scandir(path, suffix=IMG_EXTENSIONS, recursive=True)
        images = [osp.join(path, v) for v in images]
        assert images, f'{path} has no valid image file.'
        return images

    def evaluate(self, results: List[Tuple]) -> Dict:
        """Evaluating with saving generated images. (needs no metrics)

        Args:
            results (list[tuple]): The output of forward_test() of the model.

        Return:
            dict: Evaluation results dict
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result
