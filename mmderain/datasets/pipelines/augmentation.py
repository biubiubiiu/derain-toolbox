# This code is taken from https://github.com/open-mmlab/mmediting
# Modified by Raymond Wong

import copy
import math
import numbers

import cv2
import mmcv
import numpy as np

from ..registry import PIPELINES


@PIPELINES.register_module()
class Resize:
    """Resize data to a specific size for training or resize the images to fit
    the network input regulation for testing.

    When used for resizing images to fit network input regulation, the case is
    that a network may have several downsample and then upsample operation,
    then the input height and width should be divisible by the downsample
    factor of the network.
    For example, the network would downsample the input for 5 times with
    stride 2, then the downsample factor is 2^5 = 32 and the height
    and width should be divisible by 32.

    Required keys are the keys in attribute "keys", added or modified keys are
    "keep_ratio", "scale_factor", "interpolation" and the
    keys in attribute "keys".

    All keys in "keys" should have the same shape. "test_trans" is used to
    record the test transformation to align the input's shape.

    Args:
        keys (list[str]): The images to be resized.
        scale (float | Tuple[int]): If scale is Tuple(int), target spatial
            size (h, w). Otherwise, target spatial size is scaled by input
            size.
            Note that when it is used, `size_factor` and `max_size` are
            useless. Default: None
        keep_ratio (bool): If set to True, images will be resized without
            changing the aspect ratio. Otherwise, it will resize images to a
            given size. Default: False.
            Note that it is used togher with `scale`.
        size_factor (int): Let the output shape be a multiple of size_factor.
            Default:None.
            Note that when it is used, `scale` should be set to None and
            `keep_ratio` should be set to False.
        max_size (int): The maximum size of the longest side of the output.
            Default:None.
            Note that it is used togher with `size_factor`.
        interpolation (str): Algorithm used for interpolation:
            "nearest" | "bilinear" | "bicubic" | "area" | "lanczos".
            Default: "bilinear".
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used.
            Default: None.
        output_keys (list[str] | None): The resized images. Default: None
            Note that if it is not `None`, its length should be equal to keys.
    """

    def __init__(self,
                 keys,
                 scale=None,
                 keep_ratio=False,
                 size_factor=None,
                 max_size=None,
                 interpolation='bilinear',
                 backend=None,
                 output_keys=None):
        assert keys, 'Keys should not be empty.'
        if output_keys:
            assert len(output_keys) == len(keys)
        else:
            output_keys = keys
        if size_factor:
            assert scale is None, ('When size_factor is used, scale should ',
                                   f'be None. But received {scale}.')
            assert keep_ratio is False, ('When size_factor is used, '
                                         'keep_ratio should be False.')
        if max_size:
            assert size_factor is not None, (
                'When max_size is used, '
                f'size_factor should also be set. But received {size_factor}.')
        if isinstance(scale, float):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
        elif mmcv.is_tuple_of(scale, int):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            if max_short_edge == -1:
                # assign np.inf to long edge for rescaling short edge later.
                scale = (np.inf, max_long_edge)
        elif scale is not None:
            raise TypeError(
                f'Scale must be None, float or tuple of int, but got '
                f'{type(scale)}.')
        self.keys = keys
        self.output_keys = output_keys
        self.scale = scale
        self.size_factor = size_factor
        self.max_size = max_size
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.backend = backend

    def _resize(self, img):
        if self.keep_ratio:
            img, self.scale_factor = mmcv.imrescale(
                img,
                self.scale,
                return_scale=True,
                interpolation=self.interpolation,
                backend=self.backend)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img,
                self.scale,
                return_scale=True,
                interpolation=self.interpolation,
                backend=self.backend)
            self.scale_factor = np.array((w_scale, h_scale), dtype=np.float32)
        return img

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        if self.size_factor:
            h, w = results[self.keys[0]].shape[:2]
            new_h = h - (h % self.size_factor)
            new_w = w - (w % self.size_factor)
            if self.max_size:
                new_h = min(self.max_size - (self.max_size % self.size_factor),
                            new_h)
                new_w = min(self.max_size - (self.max_size % self.size_factor),
                            new_w)
            self.scale = (new_w, new_h)
        for key, out_key in zip(self.keys, self.output_keys):
            results[out_key] = self._resize(results[key])
            if len(results[out_key].shape) == 2:
                results[out_key] = np.expand_dims(results[out_key], axis=2)

        results['scale_factor'] = self.scale_factor
        results['keep_ratio'] = self.keep_ratio
        results['interpolation'] = self.interpolation
        results['backend'] = self.backend

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, output_keys={self.output_keys}, '
            f'scale={self.scale}, '
            f'keep_ratio={self.keep_ratio}, size_factor={self.size_factor}, '
            f'max_size={self.max_size},interpolation={self.interpolation})')
        return repr_str


@PIPELINES.register_module()
class Flip:
    """Flip the input data with a probability.

    Reverse the order of elements in the given data with a specific direction.
    The shape of the data is preserved, but the elements are reordered.
    Required keys are the keys in attributes "keys", added or modified keys are
    "flip", "flip_direction" and the keys in attributes "keys".
    It also supports flipping a list of images with the same flip.

    Args:
        keys (list[str]): The images to be flipped.
        flip_ratio (float): The propability to flip the images.
        direction (str): Flip images horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self, keys, flip_ratio=0.5, direction='horizontal'):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported.'
                             f'Currently support ones are {self._directions}')
        self.keys = keys
        self.flip_ratio = flip_ratio
        self.direction = direction

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        flip = np.random.random() < self.flip_ratio

        if flip:
            for key in self.keys:
                if isinstance(results[key], list):
                    for v in results[key]:
                        mmcv.imflip_(v, self.direction)
                else:
                    mmcv.imflip_(results[key], self.direction)

        results['flip'] = flip
        results['flip_direction'] = self.direction

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, flip_ratio={self.flip_ratio}, '
                     f'direction={self.direction})')
        return repr_str


@PIPELINES.register_module()
class Rotate:
    """Apply Rotate Transformation to image.

    Args:
        keys (list[str]): The images to be rotated.
        angle (int | float): Rotation angle in degrees, positive values mean
            clockwise rotation.
        rotate_ratio (float): The propability to rotate the images.
        scale (int | float): Isotropic scale factor. Same in
            ``mmcv.imrotate``.
        center (int | float | tuple[float]): Center point (w, h) of the
            rotation in the source image. If None, the center of the
            image will be used. Same in ``mmcv.imrotate``.
        img_fill_val (int | float | tuple): The fill value for image border.
            If float, the same value will be used for all the three
            channels of image. If tuple, the should be 3 elements (e.g.
            equals the number of channels for image).
    """

    def __init__(self,
                 keys,
                 angle,
                 rotate_ratio,
                 scale=1,
                 center=None,
                 img_fill_val=128):
        assert isinstance(angle, (int, float)), \
            f'The angle must be type int or float. got {type(angle)}.'
        assert isinstance(scale, (int, float)), \
            f'The scale must be type int or float. got type {type(scale)}.'
        if isinstance(center, (int, float)):
            center = (center, center)
        elif isinstance(center, tuple):
            assert len(center) == 2, 'center with type tuple must have '\
                f'2 elements. got {len(center)} elements.'
        else:
            assert center is None, 'center must be None or type int, '\
                f'float or tuple, got type {type(center)}.'
        if isinstance(img_fill_val, (float, int)):
            img_fill_val = tuple([float(img_fill_val)] * 3)
        elif isinstance(img_fill_val, tuple):
            assert len(img_fill_val) == 3, 'img_fill_val as tuple must '\
                f'have 3 elements. got {len(img_fill_val)}.'
            img_fill_val = tuple([float(val) for val in img_fill_val])
        else:
            raise ValueError(
                'img_fill_val must be float or tuple with 3 elements.')
        assert np.all([0 <= val <= 255 for val in img_fill_val]), \
            'all elements of img_fill_val should between range [0,255]. '\
            f'got {img_fill_val}.'

        self.keys = keys
        self.angle = angle
        self.rotate_ratio = rotate_ratio
        self.scale = scale
        self.center = center
        self.img_fill_val = img_fill_val

    def _rotate_img(self, img, angle):
        """Rotate the image.

        Args:
            results (dict): Result dict from loading pipeline.
            angle (float): Rotation angle in degrees, positive values
                mean clockwise rotation. Same in ``mmcv.imrotate``.
            center (tuple[float], optional): Center point (w, h) of the
                rotation. Same in ``mmcv.imrotate``.
            scale (int | float): Isotropic scale factor. Same in
                ``mmcv.imrotate``.
        """
        h, w = img.shape[:2]
        center = self.center
        if center is None:
            center = ((w - 1) * 0.5, (h - 1) * 0.5)

        img_rotated = mmcv.imrotate(
            img, angle, center, self.scale, border_value=self.img_fill_val)
        return img_rotated

    def __call__(self, results):
        """Call function to rotate images, bounding boxes, masks and semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """
        rotate = np.random.random() < self.rotate_ratio

        if rotate:
            angle = self.angle
            for key in self.keys:
                img = results[key].copy()
                results[key] = self._rotate_img(img, angle)
        else:
            angle = 0

        results['rotate'] = rotate
        results['rotate_angle'] = angle
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'keys={self.keys}, '
        repr_str += f'angle={self.angle}, '
        repr_str += f'rotate_ratio={self.rotate_ratio}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'center={self.center}, '
        repr_str += f'img_fill_val={self.img_fill_val}, '
        return repr_str


@PIPELINES.register_module()
class Pad:
    """Pad the images to align with network downsample factor for testing.

    See `Reshape` for more explanation. `numpy.pad` is used for the pad
    operation.
    Required keys are the keys in attribute "keys", added or
    modified keys are "test_trans" and the keys in attribute
    "keys". All keys in "keys" should have the same shape. "test_trans" is used
    to record the test transformation to align the input's shape.

    Args:
        keys (list[str]): The images to be padded.
        ds_factor (int): Downsample factor of the network. The height and
            weight will be padded to a multiple of ds_factor. Default: 1.
        min_size (int | float | tuple[float], optional): Minumum size of the 
            output image. Default: None.
        kwargs (option): any keyword arguments to be passed to `numpy.pad`.
    """

    def __init__(self, keys, ds_factor=1, min_size=None, **kwargs):
        if isinstance(min_size, (int, float)):
            min_size = (min_size, min_size)
        elif isinstance(min_size, tuple):
            assert len(min_size) == 2, 'center with type tuple must have '\
                f'2 elements. got {len(min_size)} elements.'
        else:
            assert min_size is None, 'min_size must be None or type int, '\
                f'float or tuple, got type {type(min_size)}.'

        self.keys = keys
        self.ds_factor = ds_factor
        self.min_size = min_size
        self.kwargs = kwargs

    def _get_padded_shape(self, h, w):
        new_h = self.ds_factor * ((h - 1) // self.ds_factor + 1)
        new_w = self.ds_factor * ((w - 1) // self.ds_factor + 1)

        if self.min_size is not None:
            min_h, min_w = self.min_size
            if new_h < min_h:
                new_h = self.ds_factor * ((min_h - 1) // self.ds_factor + 1)
            if new_w < min_w:
                new_w = self.ds_factor * ((min_w - 1) // self.ds_factor + 1)

        return new_h, new_w

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        h, w = results[self.keys[0]].shape[:2]

        new_h, new_w = self._get_padded_shape(h, w)

        pad_h = new_h - h
        pad_w = new_w - w
        if new_h != h or new_w != w:
            pad_width = ((0, pad_h), (0, pad_w), (0, 0))
            for key in self.keys:
                results[key] = np.pad(results[key],
                                      pad_width[:results[key].ndim],
                                      **self.kwargs)
        results['pad'] = (pad_h, pad_w)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        kwargs_str = ', '.join(
            [f'{key}={val}' for key, val in self.kwargs.items()])
        repr_str += (f'(keys={self.keys}, ds_factor={self.ds_factor}, '
                     f'{kwargs_str})')
        return repr_str


@PIPELINES.register_module()
class RandomAffine:
    """Random affine transformation of input images keeping center invariant

    This class is adopted from
    https://github.com/pytorch/vision/blob/v0.5.0/torchvision/transforms/transforms.py#L1015
    It should be noted that in
    https://github.com/Yaoyi-Li/GCA-Matting/blob/master/dataloader/data_generator.py#L70
    random flip is added. See explanation of `flip_ratio` below.
    Required keys are the keys in attribute "keys", modified keys
    are keys in attribute "keys".

    Args:
        keys (Sequence[str]): The images to be affined.
        degrees (sequence or float or int): Range of degrees to select from. If it
            is a float instead of a tuple like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): Tuple of maximum absolute fraction for
            horizontal and vertical translations. For example translate=(a, b),
            then horizontal shift is randomly sampled in the range
            -img_width * a < dx < img_width * a and vertical shift is randomly
            sampled in the range -img_height * b < dy < img_height * b.
            Default: None.
        scale (tuple, optional): Scaling factor interval, e.g (a, b), then
            scale is randomly sampled from the range a <= scale <= b.
            Default: None.
        shear (sequence or float or int, optional): Range of shear degrees to
            select from. If shear is a float, a shear parallel to the x axis
            and a shear parallel to the y axis in the range (-shear, +shear)
            will be applied. Else if shear is a tuple of 2 values, a x-axis
            shear and a y-axis shear in (shear[0], shear[1]) will be applied.
            Default: None.
        resample ('nearest', 'bilinear', 'bicubic'}, optional):
            An optional resampling filters. Default: 'nearest'
        flip_ratio (float, optional): Probability of the image being flipped.
            The flips in horizontal direction and vertical direction are
            independent. The image may be flipped in both directions.
            Default: None.
    """

    _str_to_cv2_interpolation = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC
    }

    def __init__(self,
                 keys,
                 degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 resample='nearest',
                 flip_ratio=None):
        self.keys = keys
        if isinstance(degrees, numbers.Number):
            assert degrees >= 0, ('If degrees is a single number, '
                                  'it must be positive.')
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                'degrees should be a list or tuple and it must be of length 2.'
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                'translate should be a list or tuple and it must be of length 2.'
            for t in translate:
                assert 0.0 <= t <= 1.0, ('translation values should be '
                                         'between 0 and 1.')
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                'scale should be a list or tuple and it must be of length 2.'
            for s in scale:
                assert s > 0, 'scale values should be positive.'
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                assert shear >= 0, ('If shear is a single number, '
                                    'it must be positive.')
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    'shear should be a list or tuple and it must be of length 2 or 4.'
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

        if flip_ratio is not None:
            assert isinstance(flip_ratio,
                              float), 'flip_ratio should be a float.'
            self.flip_ratio = flip_ratio
        else:
            self.flip_ratio = 0

        assert resample in self._str_to_cv2_interpolation, \
            'Resample options should be one of "nearest", "bilinear" and "bicubic"'

        self.resample = resample

    @staticmethod
    def _get_params(degrees, translate, scale_ranges, shears, flip_ratio,
                    img_size):
        """Get parameters for affine transformation.

        Returns:
            paras (tuple): Params to be passed to the affine transformation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(np.random.uniform(-max_dx, max_dx)),
                            np.round(np.random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (np.random.uniform(scale_ranges[0], scale_ranges[1]),
                     np.random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            if len(shears) == 2:
                shear = [np.random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [np.random.uniform(shears[0], shears[1]),
                         np.random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        # Because `flip` is used as a multiplier in line 604 and 605,
        # so -1 stands for flip and 1 stands for no flip. Thus `flip`
        # should be an 'inverse' flag as the result of the comparison.
        flip = (np.random.rand(2) > flip_ratio).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear,
                                   flip):
        """Helper method to compute inverse matrix for affine transformation.

        As it is explained in PIL.Image.rotate, we need compute INVERSE of
        affine transformation matrix: M = T * C * RSS * C^-1 where
        T is translation matrix:
            [1, 0, tx | 0, 1, ty | 0, 0, 1];
        C is translation matrix to keep center:
            [1, 0, cx | 0, 1, cy | 0, 0, 1];
        RSS is rotation with scale and shear matrix.

        It is different from the original function in torchvision.
        1. The order are changed to flip -> scale -> rotation -> shear.
        2. x and y have different scale factors.
        RSS(shear, a, scale, f) =
            [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
            [ sin(a)*scale_x*f          cos(a)*scale_y             0]
            [     0                       0                        1]
        Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1.
        """
        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                'Shear should be a single value or a tuple/list containing ' +
                f'two values. Got {shear}')

        angle = math.radians(angle)
        shear_x, shear_y = [math.radians(s) for s in shear]
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear_x) * math.cos(angle) + \
            math.sin(angle + shear_y) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear_x) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear_y) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation:
        # RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (
            -center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (
            -center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        h, w = results[self.keys[0]].shape[:2]
        params = self._get_params(self.degrees, self.translate, self.scale,
                                  self.shear, self.flip_ratio, (h, w))

        center = (w * 0.5 - 0.5, h * 0.5 - 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        interpolation_flag = self._str_to_cv2_interpolation[self.resample]
        for key in self.keys:
            results[key] = cv2.warpAffine(
                results[key],
                M, (w, h),
                flags=interpolation_flag + cv2.WARP_INVERSE_MAP)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, degrees={self.degrees}, '
                     f'translate={self.translate}, scale={self.scale}, '
                     f'shear={self.shear}, flip_ratio={self.flip_ratio})')
        return repr_str


@PIPELINES.register_module()
class RandomJitter:
    """Randomly jitter the foreground in hsv space.

    The jitter range of hue is adjustable while the jitter ranges of saturation
    and value are adaptive to the images. Side effect: the "fg" image will be
    converted to `np.float32`.
    Required keys are "fg" and "alpha", modified key is "fg".

    Args:
        hue_range (float | tuple[float]): Range of hue jittering. If it is a
            float instead of a tuple like (min, max), the range of hue
            jittering will be (-hue_range, +hue_range). Default: 40.
    """

    def __init__(self, hue_range=40):
        if isinstance(hue_range, numbers.Number):
            assert hue_range >= 0, ('If hue_range is a single number, '
                                    'it must be positive.')
            self.hue_range = (-hue_range, hue_range)
        else:
            assert isinstance(hue_range, tuple) and len(hue_range) == 2, \
                'hue_range should be a tuple and it must be of length 2.'
            self.hue_range = hue_range

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        fg, alpha = results['fg'], results['alpha']

        # convert to HSV space;
        # convert to float32 image to keep precision during space conversion.
        fg = mmcv.bgr2hsv(fg.astype(np.float32) / 255)
        # Hue noise
        hue_jitter = np.random.randint(self.hue_range[0], self.hue_range[1])
        fg[:, :, 0] = np.remainder(fg[:, :, 0] + hue_jitter, 360)

        # Saturation noise
        sat_mean = fg[:, :, 1][alpha > 0].mean()
        # jitter saturation within range (1.1 - sat_mean) * [-0.1, 0.1]
        sat_jitter = (1.1 - sat_mean) * (np.random.rand() * 0.2 - 0.1)
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        fg[:, :, 1] = sat

        # Value noise
        val_mean = fg[:, :, 2][alpha > 0].mean()
        # jitter value within range (1.1 - val_mean) * [-0.1, 0.1]
        val_jitter = (1.1 - val_mean) * (np.random.rand() * 0.2 - 0.1)
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = mmcv.hsv2bgr(fg)
        results['fg'] = fg * 255

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'hue_range={self.hue_range}'


class BinarizeImage:
    """Binarize image.

    Args:
        keys (Sequence[str]): The images to be binarized.
        binary_thr (float): Threshold for binarization.
        to_int (bool): If True, return image as int32, otherwise
            return image as float32.
    """

    def __init__(self, keys, binary_thr, to_int=False):
        self.keys = keys
        self.binary_thr = binary_thr
        self.to_int = to_int

    def _binarize(self, img):
        type_ = np.float32 if not self.to_int else np.int32
        img = (img[..., :] > self.binary_thr).astype(type_)

        return img

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for k in self.keys:
            results[k] = self._binarize(results[k])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, binary_thr={self.binary_thr}, '
                     f'to_int={self.to_int})')

        return repr_str


@PIPELINES.register_module()
class RandomMaskDilation:
    """Randomly dilate binary masks.

    Args:
        keys (Sequence[str]): The images to be resized.
        get_binary (bool): If True, according to binary_thr, reset final
            output as binary mask. Otherwise, return masks directly.
        binary_thr (float): Threshold for obtaining binary mask.
        kernel_min (int): Min size of dilation kernel.
        kernel_max (int): Max size of dilation kernel.
    """

    def __init__(self, keys, binary_thr=0., kernel_min=9, kernel_max=49):
        self.keys = keys
        self.kernel_min = kernel_min
        self.kernel_max = kernel_max
        self.binary_thr = binary_thr

    def _random_dilate(self, img):
        kernel_size = np.random.randint(self.kernel_min, self.kernel_max + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        dilate_kernel_size = kernel_size
        img_ = cv2.dilate(img, kernel, iterations=1)

        img_ = (img_ > self.binary_thr).astype(np.float32)

        return img_, dilate_kernel_size

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        for k in self.keys:
            results[k], d_kernel = self._random_dilate(results[k])
            if len(results[k].shape) == 2:
                results[k] = np.expand_dims(results[k], axis=2)
            results[k + '_dilate_kernel_size'] = d_kernel

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, kernel_min={self.kernel_min}, '
                     f'kernel_max={self.kernel_max})')

        return repr_str


@PIPELINES.register_module()
class RandomTransposeHW:
    """Randomly transpose images in H and W dimensions with a probability.

    (TransposeHW = horizontal flip + anti-clockwise rotatation by 90 degrees)
    When used with horizontal/vertical flips, it serves as a way of rotation
    augmentation.
    It also supports randomly transposing a list of images.

    Required keys are the keys in attributes "keys", added or modified keys are
    "transpose" and the keys in attributes "keys".

    Args:
        keys (list[str]): The images to be transposed.
        transpose_ratio (float): The propability to transpose the images.
    """

    def __init__(self, keys, transpose_ratio=0.5):
        self.keys = keys
        self.transpose_ratio = transpose_ratio

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict containing the processed data and information.
        """
        transpose = np.random.random() < self.transpose_ratio

        if transpose:
            for key in self.keys:
                if isinstance(results[key], list):
                    results[key] = [v.transpose(1, 0, 2) for v in results[key]]
                else:
                    results[key] = results[key].transpose(1, 0, 2)

        results['transpose'] = transpose

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f'(keys={self.keys}, transpose_ratio={self.transpose_ratio})')
        return repr_str


@PIPELINES.register_module()
class CopyValues:
    """Copy the value of a source key to a destination key.


    It does the following: results[dst_key] = results[src_key] for
    (src_key, dst_key) in zip(src_keys, dst_keys).

    Added keys are the keys in the attribute "dst_keys".

    Args:
        src_keys (list[str]): The source keys.
        dst_keys (list[str]): The destination keys.
    """

    def __init__(self, src_keys, dst_keys):

        if not isinstance(src_keys, list) or not isinstance(dst_keys, list):
            raise AssertionError('"src_keys" and "dst_keys" must be lists.')

        if len(src_keys) != len(dst_keys):
            raise ValueError('"src_keys" and "dst_keys" should have the same'
                             'number of elements.')

        self.src_keys = src_keys
        self.dst_keys = dst_keys

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with a key added/modified.
        """
        for (src_key, dst_key) in zip(self.src_keys, self.dst_keys):
            results[dst_key] = copy.deepcopy(results[src_key])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(src_keys={self.src_keys})')
        repr_str += (f'(dst_keys={self.dst_keys})')
        return repr_str


@PIPELINES.register_module()
class Quantize:
    """Quantize and clip the image to [0, 1].

    It is assumed that the the input has range [0, 1].

    Modified keys are the attributes specified in "keys".

    Args:
        keys (list[str]): The keys whose values are clipped.
    """

    def __init__(self, keys):
        self.keys = keys

    def _quantize_clip(self, input_):
        is_single_image = False
        if isinstance(input_, np.ndarray):
            is_single_image = True
            input_ = [input_]

        # quantize and clip
        input_ = [np.clip((v * 255.0).round(), 0, 255) / 255. for v in input_]

        if is_single_image:
            input_ = input_[0]

        return input_

    def __call__(self, results):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.

        Returns:
            dict: A dict with the values of the specified keys are rounded
                and clipped.
        """

        for key in self.keys:
            results[key] = self._quantize_clip(results[key])

        return results

    def __repr__(self):
        return self.__class__.__name__
