import numbers
import os.path as osp

import mmcv
import torch
from mmcv import ConfigDict

from mmderain.core import psnr, ssim, tensor2img

from ..base import BaseModel
from ..builder import build_backbone, build_loss
from ..common import set_requires_grad
from ..registry import MODELS


@MODELS.register_module()
class ECNet(BaseModel):
    """Restoration model for ECNet.

    Paper: Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM

    Args:
        rain_encoder (dict): Config for the rain-to-rain autoencoder
        derain_net (dict, optional): Config for the deraining network. Default: None.
        loss_self (dict, optional): Config for self supervised loss. Default: None.
        loss_embed (dict, optional): Config for rain embedding loss. Default: None.
        loss_att (dict, optional): Config for attention map loss. Default: None.
        train_cfg (dict): Config for training. In ``train_cfg``,
            ``train_ecnet`` should be specified. If ``train_ecnet`` is set to
            True, then ``stage_loss_weights`` and ``mask_threshold`` should be
            secified.
        test_cfg (dict): Config for testing. In ``test_cfg``,
            ``test_ecnet`` should be specfied if ecnet is integrated.
        init_cfg: (dict or list[dict], optional): Initialization config dict. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 rain_encoder,
                 derain_net=None,
                 loss_self=None,
                 loss_embed=None,
                 loss_att=None,
                 loss_image=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.train_cfg = train_cfg if train_cfg is not None else ConfigDict()
        self.test_cfg = test_cfg if test_cfg is not None else ConfigDict()

        self.rain_encoder = build_backbone(rain_encoder)
        # build derain_net if it's not None
        if derain_net is None:
            self.train_cfg['train_ecnet'] = False
            self.test_cfg['test_ecnet'] = False
        else:
            self.derain_net = build_backbone(derain_net)

        # if argument train_cfg is not None, validate if the config is proper.
        if train_cfg is not None:
            assert hasattr(self.train_cfg, 'train_ecnet')
            assert hasattr(self.test_cfg, 'test_ecnet')
            if self.test_cfg.test_ecnet and not self.train_cfg.train_ecnet:
                raise ValueError('You are not training ecnet, but it is used for '
                                 'model forwarding.')
            if self.train_cfg.train_ecnet:
                assert self.train_cfg.get('stage_loss_weights', None) is not None, \
                    'loss weights for each stage should be provided'
                assert self.train_cfg.get('mask_threshold', None) is not None, \
                    'Mask threshold should be provided'

                if all(v is None for v in (loss_embed, loss_att, loss_image)):
                    raise ValueError('Please specify one loss for ECNet')

                self.mask_threshold = self.train_cfg.mask_threshold
                self.stage_loss_weights = self.train_cfg.stage_loss_weights

                set_requires_grad(self.rain_encoder, False)  # freeze autoencoder

        self.loss_embed = build_loss(loss_embed) if loss_embed is not None else None
        self.loss_att = build_loss(loss_att) if loss_att is not None else None
        self.loss_image = build_loss(loss_image) if loss_image is not None else None
        self.loss_self = build_loss(loss_self) if loss_self is not None else None

        # support fp16
        self.fp16_enabled = False


    def forward_train(self, lq, gt):
        """Training forward function.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            Tensor: Output tensor.
        """

        R = lq-gt
        output, rain_embedding = self.rain_encoder(R)

        losses = dict()
        if not self.train_cfg.train_ecnet:
            losses['loss_self_supervised'] = self.loss_self(output, R)
        else:
            derains, embeddings, attention_maps = self.derain_net(lq)
            if self.loss_embed is not None:
                losses['loss_embedding'] = [self.loss_embed(em, rain_embedding)
                                            for em in embeddings]
            if self.loss_att is not None:
                rain_mask = (R > self.mask_threshold).type(R.dtype)
                rain_mask, _ = torch.max(rain_mask, dim=1, keepdim=True)
                losses['loss_attention'] = [self.loss_att(att, rain_mask)
                                            for att in attention_maps]
            if self.loss_image is not None:
                losses['loss_reconstruct'] = [self.loss_image((out+1)/2, (gt+1)/2)  # denormalize
                                              for out in derains]

            # calculate weighted sum of losses at each stage
            for key, loss in losses.items():
                assert len(self.stage_loss_weights) == len(loss)
                losses[key] = sum([val*w for val, w in zip(loss, self.stage_loss_weights)])

        outputs = dict(
            losses=losses,
            num_samples=len(gt.data),
            results=dict(lq=lq.cpu(), gt=gt.cpu(), output=output.cpu()))
        return outputs

    def forward_test(self,
                     lq,
                     gt=None,
                     lq_unnormalised=None,
                     gt_unnormalised=None,
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
        if self.test_cfg.test_ecnet:
            derains, _, _ = self.derain_net(lq)
            output = derains[-1]
        else:
            output, _ = self.rain_encoder(lq-gt)

        output = (output+1)/2  # denormalize

        if meta is not None and 'pad' in meta[0]:
            output = self.restore_shape(output, meta)
            lq = self.restore_shape(lq, meta)
            gt = self.restore_shape(gt, meta)
            lq_unnormalised = self.restore_shape(lq_unnormalised, meta)
            gt_unnormalised = self.restore_shape(gt_unnormalised, meta)

        if self.test_cfg.test_ecnet:
            label = gt_unnormalised
        else:
            label = lq_unnormalised - gt_unnormalised

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert label is not None, ('evaluation with metrics must have gt images.')
            results = dict(eval_result=self.evaluate(output, label))
        else:
            results = dict(output=output.cpu())
            if self.test_cfg.test_ecnet:
                results['lq'] = lq.cpu()
            if label is not None:
                results['gt'] = label.cpu()

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
        if self.test_cfg.test_ecnet:
            out = self.derain_net(img)
        else:
            out = self.rain_encoder(img)
        return out
