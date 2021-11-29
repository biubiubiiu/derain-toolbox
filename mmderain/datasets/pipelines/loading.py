# This code is taken from https://github.com/open-mmlab/mmediting
# Modified by Raymond Wong

import warnings
import mmcv
import numpy as np
from mmcv.fileio import FileClient

from ..registry import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Load image from file.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        filepath (str): The path to find the images.
        key (str): Keys in results. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        use_cache (bool): If True, load all images at once. Default: False.
        backend (str): The image loading backend type. Options are `cv2`,
            `pillow`, and 'turbojpeg'. Default: None.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 filepath,
                 io_backend='disk',
                 key='gt',
                 flag='color',
                 channel_order='bgr',
                 save_original_img=False,
                 use_cache=False,
                 backend=None,
                 **kwargs):
        self.filepath = filepath
        self.io_backend = io_backend
        self.key = key
        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.kwargs = kwargs
        self.file_client = None
        self.use_cache = use_cache
        self.cache = None
        self.backend = backend

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        if self.use_cache:
            if self.cache is None:
                self.cache = dict()
            if self.filepath in self.cache:
                img = self.cache[self.filepath]
            else:
                img_bytes = self.file_client.get(self.filepath)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.flag,
                    channel_order=self.channel_order,
                    backend=self.backend)  # HWC
                self.cache[self.filepath] = img
        else:
            img_bytes = self.file_client.get(self.filepath)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.flag,
                channel_order=self.channel_order,
                backend=self.backend)  # HWC
        results[self.key] = img
        results[f'{self.key}_path'] = self.filepath
        results[f'{self.key}_ori_shape'] = img.shape
        if self.save_original_img:
            results[f'ori_{self.key}'] = img.copy()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(io_backend={self.io_backend}, key={self.key}, '
            f'flag={self.flag}, save_original_img={self.save_original_img}, '
            f'channel_order={self.channel_order}, use_cache={self.use_cache})')
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileList(LoadImageFromFile):
    """Load image from file list.

    It accepts a list of path and read each frame from each path. A list
    of frames will be returned.

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        key (str): Keys in results to find corresponding path. Default: 'gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        filepaths = results[f'{self.key}_path']
        if not isinstance(filepaths, list):
            raise TypeError(
                f'filepath should be list, but got {type(filepaths)}')

        filepaths = [str(v) for v in filepaths]

        imgs = []
        shapes = []
        if self.save_original_img:
            ori_imgs = []
        for filepath in filepaths:
            img_bytes = self.file_client.get(filepath)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.flag,
                channel_order=self.channel_order)  # HWC
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            imgs.append(img)
            shapes.append(img.shape)
            if self.save_original_img:
                ori_imgs.append(img.copy())

        results[self.key] = imgs
        results[f'{self.key}_path'] = filepaths
        results[f'{self.key}_ori_shape'] = shapes
        if self.save_original_img:
            results[f'ori_{self.key}'] = ori_imgs

        return results


@PIPELINES.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """Load a pair of images from file.

    Each sample contains a pair of images, which are concatenated in the w
    dimension (a|b). This is a special loading class for generation paired
    dataset. It loads a pair of images as the common loader does and crops
    it into two images with the same shape in different domains.

    Required key is "filepath". Added keys are "pair",
    "pair_path", "pair_ori_shape", "{key_a}", "{key_b}", "{key_a}_path",
    "{key_b}_path", "{key_a}_ori_shape" and "{key_b}_ori_shape", where `key_a`
    and `key_b` are specified in `keys`, seperated by ','

    Args:
        io_backend (str): io backend where images are store. Default: 'disk'.
        keys (str): Keys to each part of the paired image, separated by ','. Default: 'lq,gt'.
        flag (str): Loading flag for images. Default: 'color'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
            Default: 'bgr'.
        save_original_img (bool): If True, maintain a copy of the image in
            `results` dict with name of `f'ori_{key}'`. Default: False.
        kwargs (dict): Args for file client.
    """

    def __init__(self,
                 io_backend='disk',
                 key='lq,gt',
                 flag='color',
                 channel_order='bgr',
                 save_original_img=False,
                 use_cache=False,
                 backend=None,
                 **kwargs):
        self.io_backend = io_backend
        self.keys = key.split(',')
        assert len(self.keys) == 2
        if any(key_name == 'pair' for key_name in self.keys):
            warnings.warn(
                "'pair' is reserved name in keys, be careful of name shadowing")

        self.flag = flag
        self.save_original_img = save_original_img
        self.channel_order = channel_order
        self.kwargs = kwargs
        self.file_client = None
        self.use_cache = use_cache
        self.cache = None
        self.backend = backend

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        filepath = str(results['pair_path'])
        img_bytes = self.file_client.get(filepath)
        img = mmcv.imfrombytes(
            img_bytes, flag=self.flag, channel_order=self.channel_order)  # HWC
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        results['pair'] = img
        results['pair_path'] = filepath
        results['pair_ori_shape'] = img.shape

        # crop pair into a and b
        w = img.shape[1]
        if w % 2 != 0:
            raise ValueError(
                f'The width of image pair must be even number, but got {w}.')
        new_w = w // 2
        img_a = img[:, :new_w, :]
        img_b = img[:, new_w:, :]

        for (key, img) in zip(self.keys, (img_a, img_b)):
            results[key] = img
            results[f'{key}_path'] = filepath
            results[f'{key}_ori_shape'] = img.shape
            if self.save_original_img:
                results[f'ori_img_{key}'] = img.copy()

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(io_backend={self.io_backend}, key={self.keys}, '
            f'flag={self.flag}, save_original_img={self.save_original_img}, '
            f'channel_order={self.channel_order}, use_cache={self.use_cache})')
        return repr_str
