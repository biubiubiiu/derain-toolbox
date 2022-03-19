from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class RLNetHyperParamAdjustmentHook(Hook):
    """Update the hyperparameters in RLNet training

    Specifically, this hook adjust the value of \\theta_{2}, \\lambda_{2}
    in the training process. In the first stage, \\theta_{2} is set as 0.15
    when reached 20 epochs, \\lambda_{2} is set to 6 when reached 30 epochs.
    In the second stage, \\lambda_{2} is set to 0 when reaching 30 \times
    K(K=1,2,3,4,5,6) epochs and set to 0.6 when reaching 30 \times K + 15 epochs
    (K=1,2,3,4,5,6).
    """

    def __init__(self, mode='official_code'):
        super().__init__()
        self.mode = mode

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        joint_training = model.train_cfg.get('joint_training', False)
        if self.mode == 'official_code':
            self._call_func_from_official_code(model, epoch, joint_training)
        elif self.mode == 'paper':
            self._call_func_from_paper(model, epoch, joint_training)
        else:
            raise ValueError(f'Invalid value: mode={self.mode}')

    def _call_func_from_official_code(self, model, epoch, joint_training):
        if joint_training:
            if (epoch // 15) % 2 == 0:
                model.loss_weight[2] = 0
            else:
                model.loss_weight[2] = model.lambdas[2]

    def _call_func_from_paper(self, model, epoch, joint_training):
        if joint_training:
            if epoch < 30 or (epoch // 15) % 2 == 0:
                model.loss_weight[2] = 0
            else:
                model.loss_weight[2] = model.lambdas[2]
        else:
            if epoch >= 20:
                model.generator.theta2 = 0.15
            if epoch >= 30:
                model.loss_weight[2] = 6
