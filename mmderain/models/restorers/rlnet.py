import copy
import numbers
import os.path as osp

import mmcv
import torch
import torch.nn.functional as F
from mmcv import ConfigDict

from mmderain.core import psnr, ssim, tensor2img

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..common import set_requires_grad
from ..registry import MODELS


@MODELS.register_module()
class RLNet(BaseModel):
    """Restorer for RLNet.

    Paper: Robust Representation Learning with Feedback for Single Image Deraining

    Args:
        model_cfg (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        structural_loss (dict): Config for structural loss.
        loss_weight (tuple[float]): Loss weight for intermediate output and final output
        train_cfg (dict): Config of training. In ``train_cfg``,
            ``joint_training`` should be specified. Default: None.
        test_cfg (dict): Config of testing. Default: None.
        init_cfg: (dict or list[dict], optional): Initialization config dict. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 model_cfg,
                 pixel_loss,
                 structural_loss,
                 lambdas,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.train_cfg = train_cfg if train_cfg is not None else ConfigDict()
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(model_cfg)

        joint_training = self.train_cfg.get('joint_training', False)
        if not joint_training:
            self.freeze_backbone()

        self.lambdas = lambdas

        # loss
        self.loss_weight = copy.deepcopy(lambdas)
        self.pixel_loss = build_loss(pixel_loss)
        self.structure_loss = build_loss(structural_loss)

    def freeze_backbone(self):
        """Freeze the backbone and only train the error detector
        and feature compensator
        """
        print('>>> Freeze backbone of RLNet')
        for name, module in self.generator.named_children():
            if name not in ['fc1_internel', 'fc2_internel', 'fc1_externel', 'fc2_externel',
                            'ed1', 'ed2']:
                set_requires_grad(module, False)


    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """
        R = lq-gt
        out, F1, F2, phi1, phi, y2, y4, k2, k4 = self.generator(lq, R)
        B_estimate = lq-out

        R_2 = F.avg_pool2d(R, kernel_size=2)

        lambda0, lambda1, lambda2, lambda3 = self.loss_weight

        losses = dict()

        losses['loss_e1'] = lambda1 * self.pixel_loss(phi1, R_2)
        if lambda2 > 0:
            losses['loss_e2'] = lambda2 * self.pixel_loss(
                phi, torch.div(self.generator.theta1, torch.abs(R_2-phi1)).detach())
        losses['loss_c0.25'] = lambda3 * self.pixel_loss(F1, y2)
        losses['loss_c0.5'] = lambda3 * self.pixel_loss(F2, y4)
        losses['loss_p'] = lambda0 * torch.mean(k2*k2) + lambda0 * torch.mean(k4*k4)
        losses['loss_f'] = self.pixel_loss(out, R)
        losses['loss_ssim'] = self.structure_loss(B_estimate, gt)

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=B_estimate.cpu()))
        return outputs

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
        out, _, _, _, _ = self.generator(lq)
        output = lq-out

        if meta is not None and 'pad' in meta[0]:
            output = self.restore_shape(output, meta)
            lq = self.restore_shape(lq, meta)

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, (
                'evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, gt))
        else:
            results = dict(lq=lq.cpu(), output=output.cpu())
            if gt is not None:
                results['gt'] = gt.cpu()

        # save image
        if save_image:
            lq_path = meta[0]['lq_path']
            folder_name = osp.splitext(osp.basename(lq_path))[0]
            if isinstance(iteration, numbers.Number):
                save_path = osp.join(save_path, folder_name,
                                     f'{folder_name}-{iteration + 1:06d}.png')
            elif iteration is None:
                save_path = osp.join(save_path, f'{folder_name}.png')
            else:
                raise ValueError('iteration should be number or None, '
                                 f'but got {type(iteration)}')
            save_img = mmcv.rgb2bgr(tensor2img(output))
            mmcv.imwrite(save_img, save_path)

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Input image.

        Returns:
            Tensor: Output image.
        """
        out, _, _, _, _ = self.generator(img)
        return out
