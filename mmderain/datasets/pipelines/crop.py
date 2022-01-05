# This code is taken from https://github.com/open-mmlab/mmediting
# Modified by Raymond Wong

from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np

from ..registry import PIPELINES


class Crop(object, metaclass=ABCMeta):
    """Base class for cropping data for training

    Args:
        keys (Sequence[str]): The images to be cropped.
    """

    def __init__(self, keys):
        self.keys = keys

    def _crop(self, data, x_offset, y_offset, crop_w, crop_h):
        crop_bbox = [x_offset, y_offset, crop_w, crop_h]
        data_ = data[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w,
                     ...]
        return data_, crop_bbox

    @abstractmethod
    def _get_crop_size(self, results):
        """Determine size of cropping box

        Returns:
            tuple: Target spatial size (h, w).
        """

    @abstractmethod
    def _get_crop_pos(self, results):
        """Determine cropping position (x, y) of given data

        Returns:
            tuple: Specific position of cropping location.
        """

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        crop_h, crop_w = self._get_crop_size(results)
        x_offset, y_offset = self._get_crop_pos(results)

        for k in self.keys:
            data_h, data_w = results[k].shape[:2]
            if crop_h > data_h or crop_w > data_w:
                raise ValueError(
                    'The size of the crop box exceeds the size of the image.'
                    f'crop box size: ({crop_w},{crop_h}), image size: ({data_w}, {data_h})')

            data_, crop_bbox = self._crop(results[k], x_offset, y_offset, crop_w, crop_h)
            results[k] = data_
            results[k + '_crop_bbox'] = crop_bbox
        results['crop_size'] = (crop_h, crop_w)
        results['crop_pos'] = (x_offset, y_offset)
        return results


@PIPELINES.register_module()
class RandomCrop(Crop):
    """Crop the data at random location.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
    """

    def __init__(self, keys, crop_size):
        super(RandomCrop, self).__init__(keys)

        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')

        self.crop_size = crop_size

    def _get_crop_size(self, _):
        return self.crop_size

    def _get_crop_pos(self, results):
        data_h, data_w = results[self.keys[0]].shape[:2]
        crop_h, crop_w = self.crop_size

        x_offset = np.random.randint(0, data_w - crop_w + 1)
        y_offset = np.random.randint(0, data_h - crop_h + 1)

        return x_offset, y_offset

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, ')
        return repr_str


@PIPELINES.register_module()
class CenterCrop(Crop):
    """Crop the data at the center

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
    """

    def __init__(self, keys, crop_size):
        super(CenterCrop, self).__init__(keys)

        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')

        self.crop_size = crop_size

    def _get_crop_size(self, _):
        return self.crop_size

    def _get_crop_pos(self, results):
        data_h, data_w = results[self.keys[0]].shape[:2]
        crop_h, crop_w = self.crop_size

        x_offset = max(0, (data_w - crop_w)) // 2
        y_offset = max(0, (data_h - crop_h)) // 2

        return x_offset, y_offset

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, ')
        return repr_str


@PIPELINES.register_module()
class FixedCrop(Crop):
    """Crop paired data at a specific position.

    Args:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        crop_pos (Tuple[int]): Specific position (x, y).
    """

    def __init__(self, keys, crop_size, crop_pos):
        super(FixedCrop, self).__init__(keys)

        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')

        if not mmcv.is_tuple_of(crop_pos, int) and (crop_pos is not None):
            raise TypeError(
                'Elements of crop_pos must be int and crop_pos must be'
                f' tuple or None, but got {type(crop_pos[0])} in '
                f'{type(crop_pos)}')

        self.crop_size = crop_size
        self.crop_pos = crop_pos

    def _get_crop_size(self, _):
        return self.crop_size

    def _get_crop_pos(self, _):
        return self.crop_pos

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, '
                     f'crop_pos={self.crop_pos}')
        return repr_str


@PIPELINES.register_module()
class ArgsCrop(Crop):
    """Crop the data, where the cropping instruction is specified
    in data dict arguments.

    It assumes that crop position is provided in
    `results['args']['crop_pos]` and the target size is provided
    in `results['args']['crop_size']
    """

    def _get_crop_pos(self, results):
        return results['args'].pop('crop_pos')

    def _get_crop_size(self, results):
        return results['args'].pop('crop_size')

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}')
        return repr_str
