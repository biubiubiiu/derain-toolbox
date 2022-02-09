import numbers
import os.path as osp

import mmcv
import torch
from mmcv.runner import auto_fp16

from mmderain.core import crop_border, psnr, ssim, tensor2img

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..registry import MODELS


@MODELS.register_module()
class MultiStageRestorer(BaseModel):
    """Multi-stage model for image restoration.

    It must contain a generator that takes an image as inputs and outputs a
    restored image. It also has a pixel-wise loss for training.

    The subclasses should overwrite the function `forward_train`,
    `forward_test` and `train_step`.

    Args:
        generator (dict): Config for the generator structure.
        losses (List[dict]): A list of configs for building losses.
        recurrent_loss (bool): If `True`, deploy losses on every stage. Else,
            deploy only on the last stage. Default: `True`
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        pretrained (str): Path for pretrained model. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 losses,
                 recurrent_loss=True,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # support fp16
        self.fp16_enabled = False

        # generator
        self.generator = build_backbone(generator)
        self.init_weights(pretrained)

        # loss
        self.loss = dict()
        for loss in losses:
            self.loss[loss['type'].lower()] = build_loss(loss)
        self.recurrent_loss = recurrent_loss

    def init_weights(self, pretrained=None):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
        """
        self.generator.init_weights(pretrained)

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

        outputs = [_restore_shape(it, meta) for it in outputs]
        return outputs

    @auto_fp16(apply_to=('lq', ))
    def forward(self, lq, gt=None, test_mode=False, **kwargs):
        """Forward function.

        Args:
            lq (Tensor): Input lq images.
            gt (Tensor): Ground-truth image. Default: None.
            test_mode (bool): Whether in test mode or not. Default: False.
            kwargs (dict): Other arguments.
        """

        if test_mode:
            return self.forward_test(lq, gt, **kwargs)

        return self.forward_train(lq, gt)

    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            List[Tensor]: Output tensors at each stage.
        """
        losses = dict()
        outputs = self.generator(lq)

        if outputs is torch.Tensor:
            outputs = (outputs,)

        for name, loss in self.loss.items():
            if self.recurrent_loss:
                losses[name] = torch.sum(torch.stack(
                    [loss(output, gt) for output in outputs], dim=0), dim=0)
            else:
                losses[name] = loss(outputs[-1], gt)

        ret = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(),
                         output=[t.cpu() for t in outputs])
        )
        return ret

    def evaluate(self, outputs, gt):
        """Evaluation function.

        Args:
            outputs (List[Tensor]): Model outputs with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        outputs = [tensor2img(t) for t in outputs]
        gt = tensor2img(gt)

        if hasattr(self.test_cfg, 'crop_border'):
            outputs = [crop_border(output, self.test_cfg.crop_border)
                       for output in outputs]
            gt = crop_border(gt, self.test_cfg.crop_border)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            for stage, output in enumerate(outputs):
                eval_result[f'{metric}_stage_{stage+1}'] = \
                    self.allowed_metrics[metric](output, gt)

        return eval_result

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Testing forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            save_image (bool): Whether to save image. Default: False.
            save_path (str): Path to save image. Default: None.
            iteration (int): Iteration for the saving image name.
                Default: None.

        Returns:
            dict: Output results.
        """
        outputs = self.generator(lq)

        if outputs is torch.Tensor:
            outputs = [outputs]

        if meta is not None and 'pad' in meta[0]:
            outputs = self.restore_shape(outputs, meta)
            lq = self.restore_shape(lq, meta)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(outputs, gt))
        else:
            results = dict(lq=lq.cpu(), output=[t.cpu() for t in outputs])
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            for stage, output in enumerate(outputs):
                if isinstance(iteration, numbers.Number):
                    path = osp.join(save_path, folder_name,
                                    f'{folder_name}-stage{stage+1}-{iteration + 1:06d}.png')
                elif iteration is None:
                    path = osp.join(
                        save_path, f'{folder_name}-stage{stage+1}.png')
                else:
                    raise ValueError('iteration should be number or None, '
                                     f'but got {type(iteration)}')
                mmcv.imwrite(tensor2img(output), path)

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out = self.generator(img)[-1]
        return out

    def train_step(self, data_batch, optimizer):
        """Train step.

        Args:
            data_batch (dict): A batch of data.
            optimizer (obj): Optimizer.

        Returns:
            dict: Returned output.
        """
        outputs = self(**data_batch, test_mode=False)
        loss, log_vars = self.parse_losses(outputs.pop('losses'))

        # optimize
        optimizer['generator'].zero_grad()
        loss.backward()
        optimizer['generator'].step()

        outputs.update({'log_vars': log_vars})
        return outputs

    def val_step(self, data_batch, **kwargs):
        """Validation step.

        Args:
            data_batch (dict): A batch of data.
            kwargs (dict): Other arguments for ``val_step``.

        Returns:
            dict: Returned output.
        """
        output = self.forward_test(**data_batch, **kwargs)
        return output
