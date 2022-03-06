from mmcv.cnn import INITIALIZERS
from mmcv.cnn.utils.weight_init import update_init_info
from mmcv.runner import _load_checkpoint_with_prefix
from mmcv.runner.checkpoint import get_state_dict, load_state_dict
from mmcv.utils import get_logger, print_log


@INITIALIZERS.register_module(name='ECNetTransfer')
class ECNetInitializer(object):
    """Initialize ECNet' decoder and last convolution from pretrained autoencoder"""

    def __init__(self,
                 checkpoint,
                 rain_ae_prefix='rain_encoder.',
                 prefixs_of_transferred_modules=('up_path.', 'conv_last.')):

        self.checkpoint = checkpoint
        self.rain_ae_prefix = rain_ae_prefix
        self.prefixs_of_transferred_modules = prefixs_of_transferred_modules

    def __call__(self, module):
        logger = get_logger('mmcv')
        print_log(f'load model from: {self.checkpoint}', logger=logger)

        rain_ae_state_dict = _load_checkpoint_with_prefix(self.rain_ae_prefix, self.checkpoint)

        weights_to_update = dict((key, val) for key, val in rain_ae_state_dict.items()
                                 if any(key.startswith(it)
                                        for it in self.prefixs_of_transferred_modules))
        curr_state_dict = get_state_dict(module)
        curr_state_dict.update(weights_to_update)
        load_state_dict(module, curr_state_dict, strict=True)

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self):
        info = f'{self.__class__.__name__}: initialized with {self.checkpoint}'
        return info
