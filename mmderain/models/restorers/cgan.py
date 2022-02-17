import os.path as osp
import numbers

import mmcv
import torch
from mmcv.runner import auto_fp16
from mmderain.core import crop_border, psnr, ssim, tensor2img

from ..base import BaseModel
from ..builder import build_backbone, build_component, build_loss
from ..common import set_requires_grad
from ..registry import MODELS


@MODELS.register_module()
class IDCGAN(BaseModel):
    """ID-CGAN model for single image deraining.

    Paper: Image De-Raining Using a Conditional Generative Adversarial Network

    Args:
        generator (dict): Config for the generator.
        discriminator (dict): Config for the discriminator.
        gan_loss (dict): Config for the gan loss.
        pixel_loss (dict): Config for the pixel loss. Default: None.
        perceptual_loss (dict): Config for the perceptual loss. Default: None.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg: (dict or list[dict], optional): Initialization config dict. Default: None.
    """
    allowed_metrics = {'PSNR': psnr, 'SSIM': ssim}

    def __init__(self,
                 generator,
                 discriminator,
                 gan_loss,
                 pixel_loss=None,
                 perceptual_loss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # generator
        self.generator = build_backbone(generator)
        # discriminator
        self.discriminator = build_component(discriminator)

        # losses
        assert gan_loss is not None  # gan loss cannot be None
        self.gan_loss = build_loss(gan_loss)
        self.pixel_loss = build_loss(pixel_loss) if pixel_loss else None
        self.perceptual_loss = build_loss(perceptual_loss) if perceptual_loss else None

    @auto_fp16(apply_to=('lq', ))
    def forward_train(self, lq, gt):
        """Forward function for training.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Dict of forward results for training.
        """
        fake = self.generator(lq)
        results = dict(real_B=gt, fake_B=fake, real_O=lq)
        return results

    def forward_test(self,
                     lq,
                     gt=None,
                     meta=None,
                     save_image=False,
                     save_path=None,
                     iteration=None):
        """Forward function for testing.

        Args:
            lq (Tensor): LQ Tensor with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w). Default: None.
            meta (list[dict]): Input meta data.
            save_image (bool, optional): If True, results will be saved as
                images. Default: False.
            save_path (str, optional): If given a valid str path, the results
                will be saved in this path. Default: None.
            iteration (int, optional): Iteration number. Default: None.

        Returns:
            dict: Dict of forward and evaluation results for testing.
        """

        fake = self.generator(lq)
        results = dict(real_B=gt.cpu(), fake_B=fake.cpu(), real_O=lq.cpu())

        if self.test_cfg is not None and self.test_cfg.get('metrics', None):
            assert gt is not None, ('evaluation with metrics must have gt images.')
            results.update(dict(eval_result=self.evaluate(fake, gt)))

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
            flag = mmcv.imwrite(tensor2img(fake), save_path)
            results['saved_flag'] = flag

        return results

    def forward_dummy(self, img):
        """Used for computing network FLOPs.

        Args:
            img (Tensor): Dummy input used to compute FLOPs.

        Returns:
            Tensor: Dummy output produced by forwarding the dummy input.
        """
        out = self.generator(img)
        return out

    def backward_discriminator(self, outputs):
        """Backward function for the discriminator.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        # GAN loss for the discriminator
        losses = dict()
        # conditional GAN
        fake_ab = torch.cat((outputs['real_O'], outputs['fake_B']), dim=1)
        fake_pred = self.discriminator(fake_ab.detach())
        losses['loss_gan_d_fake'] = self.gan_loss(
            fake_pred, target_is_real=False, is_disc=True)
        real_ab = torch.cat((outputs['real_O'], outputs['real_B']), dim=1)
        real_pred = self.discriminator(real_ab)
        losses['loss_gan_d_real'] = self.gan_loss(
            real_pred, target_is_real=True, is_disc=True)

        loss_d, log_vars_d = self.parse_losses(losses)
        loss_d *= 0.5
        loss_d.backward()
        return log_vars_d

    def backward_generator(self, outputs):
        """Backward function for the generator.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Loss dict.
        """
        losses = dict()
        # GAN loss for the generator
        fake_ab = torch.cat((outputs['real_O'], outputs['fake_B']), dim=1)
        fake_pred = self.discriminator(fake_ab)
        losses['loss_gan_g'] = self.gan_loss(fake_pred, target_is_real=True, is_disc=False)

        if self.pixel_loss:  # pixel loss for the generator
            losses['loss_pixel'] = self.pixel_loss(outputs['fake_B'], outputs['real_B'])
        if self.perceptual_loss:  # perceptual loss for the generator
            losses['loss_perceptual'] = self.perceptual_loss(outputs['fake_B'], outputs['real_B'])

        loss_g, log_vars_g = self.parse_losses(losses)
        loss_g.backward()
        return log_vars_g

    def train_step(self, data_batch, optimizer):
        """Training step function.

        In this function, the restorer will finish the train step following
        the pipeline:

            1. get fake image
            2. optimize discriminator
            3. optimize generator

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generator and discriminator.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # forward generator
        outputs = self(**data_batch, test_mode=False)

        log_vars = dict()

        # update discriminator
        set_requires_grad(self.discriminator, True)
        optimizer['discriminator'].zero_grad()
        log_vars.update(self.backward_discriminator(outputs=outputs))
        optimizer['discriminator'].step()

        # update generator, no updates to discriminator parameters.
        set_requires_grad(self.discriminator, False)
        optimizer['generator'].zero_grad()
        log_vars.update(self.backward_generator(outputs=outputs))
        optimizer['generator'].step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs['fake_B']),
            results=dict(
                real_O=outputs['real_O'].cpu(),
                fake_B=outputs['fake_B'].cpu(),
                real_B=outputs['real_B'].cpu()))

        return results

    def evaluate(self, output, gt):
        """Evaluation function.

        Args:
            output (Tensor): Model output with shape (n, c, h, w).
            gt (Tensor): GT Tensor with shape (n, c, h, w).

        Returns:
            dict: Evaluation results.
        """
        output = tensor2img(output)
        gt = tensor2img(gt)

        if hasattr(self.test_cfg, 'crop_border'):
            output = crop_border(output, self.test_cfg.crop_border)
            gt = crop_border(gt, self.test_cfg.crop_border)

        eval_result = dict()
        for metric in self.test_cfg.metrics:
            eval_result[metric] = self.allowed_metrics[metric](output, gt)

        return eval_result
