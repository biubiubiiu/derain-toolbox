# This code is taken from https://github.com/open-mmlab/mmdetection
# Modified by Raymond Wong

import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(BaseEvalHook):
    """Non-Distributed evaluation hook .

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.
    """

    def __init__(self, *args, **eval_kwargs):
        super(EvalHook, self).__init__(*args, **eval_kwargs)
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)

    def _do_evaluate(self, runner):
        """perform evaluation"""
        if not self._should_evaluate(runner):
            return

        from mmderain.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.epoch if self.by_epoch else runner.iter)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Evaluation function.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            results (dict): Model forward results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(BaseDistEvalHook):
    """Distributed evaluation hook."""

    def __init__(self, *args, **kwargs):
        super(DistEvalHook, self).__init__(*args, **kwargs)
        self.save_image = self.eval_kwargs.pop('save_image', False)
        self.save_path = self.eval_kwargs.pop('save_path', None)

    def _do_evaluate(self, runner):
        """perform evaluation"""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for _, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        from mmderain.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            gpu_collect=self.gpu_collect,
            save_image=self.save_image,
            save_path=self.save_path,
            iteration=runner.epoch if self.by_epoch else runner.iter)

        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Evaluation function.

        Args:
            runner (``mmcv.runner.BaseRunner``): The runner.
            results (dict): Model forward results.
        """
        eval_res = self.dataloader.dataset.evaluate(
            results, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
