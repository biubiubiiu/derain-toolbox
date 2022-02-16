from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Sequence

import torch
from mmcv.runner import BaseModule, auto_fp16


class BaseModel(BaseModule, metaclass=ABCMeta):
    """Base model.

    All models should subclass it.
    All subclass should overwrite:
        ``forward_train``, supporting to forward when training.

        ``forward_test``, supporting to forward when testing.
    """

    def __init__(self, init_cfg=None):
        super(BaseModel, self).__init__(init_cfg)

        # support fp16
        self.fp16_enabled = False

    @abstractmethod
    def forward_train(self, lq, gt):
        """Abstract method for training forward.

        All subclass should overwrite it.
        """

    @abstractmethod
    def forward_test(self, lq, gt=None):
        """Abstract method for testing forward.

        All subclass should overwrite it.
        """

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function. Calls either :func:`forward_train` or
        :func:`forward_test` depending on whether ``test_mode`` is ``True``

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def train_step(self, data_batch, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): A batch of data.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        outputs.update({'loss': loss, 'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """The iteration step during validation.

        This method shares a similar signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output

    def parse_losses(self, losses):
        """Parse losses dict for different loss variants.

        Args:
            losses (dict): Loss dict.

        Returns:
            loss (float): Sum of the total loss.
            log_vars (dict): loss dict for different variants.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for name in log_vars:
            log_vars[name] = log_vars[name].item()

        return loss, log_vars

    def restore_shape(self, outputs, meta):
        """Restore the predicted images to the original shape.

        Args:
            pred (Tensor): The predicted tensor with shape (n, c, h, w).
            meta (list[dict]): Meta data about the current data batch.
                Currently only batch_size 1 is supported.

        Returns:
            np.ndarray: The reshaped predicted image.
        """

        def _restore_shape(pred, meta):
            ori_h, ori_w = meta[0]['lq_ori_shape'][:2]
            pred = pred[:, :, :ori_h, :ori_w]
            return pred

        if outputs is torch.Tensor:
            return _restore_shape(outputs, meta)
        elif outputs is Sequence:
            return [_restore_shape(it, meta) for it in outputs]
        else:
            raise TypeError(f'Unexpected type of outputs: {type(outputs)}')
