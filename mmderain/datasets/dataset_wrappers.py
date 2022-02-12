import itertools
from collections import defaultdict

from mmcv.utils import is_tuple_of, to_2tuple
from PIL import Image

from .registry import DATASETS


@DATASETS.register_module()
class ExhaustivePatchDataset:
    """A wrapper of dataset that crops images into multiple patches

    Originally used in PReNet (CVPR' 2018)

    Args:
        dataset (:obj:`Dataset`): The dataset to extract patches.
        patch_size (int or tuple): Size of image patch.
        stride (int or tuple): Stride of the patches.
    """

    def __init__(self, dataset, patch_size, stride):
        self.dataset = dataset

        if not isinstance(patch_size, int) or is_tuple_of(patch_size, int):
            raise TypeError(f'patch size must be int or tuple, but got {type(patch_size)}')
        if not isinstance(stride, int) or is_tuple_of(stride, int):
            raise TypeError(f'stride must be int or tuple, but got {type(stride)}')

        self.patch_size = to_2tuple(patch_size)
        self.stride = to_2tuple(stride)

        self.patch_infos = self._get_patch_infos()

    def _retrieve_data_info(self):
        if self.dataset.test_mode:
            prepare_data = self.dataset.prepare_test_data
        else:
            prepare_data = self.dataset.prepare_train_data

        data_infos = [prepare_data(idx) for idx in range(len(self.dataset))]
        return data_infos

    def _get_patch_infos(self):
        data_infos = self._retrieve_data_info()
        patch_infos = []
        for idx, data in enumerate(data_infos):
            paths = [value for key, value in data.items() if 'path' in key.lower()]
            is_paired = len(paths) == 1

            images = [Image.open(path) for path in paths]
            if any(img.size != images[0].size for img in images):
                raise ValueError('Images should have the same size')

            w, h = images[0].size
            if is_paired:
                w = w // 2

            start_xs = list(range(0, w-self.patch_size[0]+1, self.stride[0]))
            start_ys = list(range(0, h-self.patch_size[1]+1, self.stride[1]))
            top_left_anchors = list(itertools.product(start_xs, start_ys))
            anchors_with_img_idx = list(itertools.product((idx,), top_left_anchors))
            patch_infos.extend(anchors_with_img_idx)

        return patch_infos

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        raw_data_idx, crop_pos = self.patch_infos[idx]
        if self.dataset.test_mode:
            data = self.dataset.prepare_test_data(raw_data_idx)
        else:
            data = self.dataset.prepare_train_data(raw_data_idx)

        # inject information of crop position
        data['args'] = {
            'crop_pos': crop_pos,
            'crop_size': self.patch_size
        }

        return self.dataset.pipeline(data)

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.patch_infos)

    def evaluate(self, results):
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
