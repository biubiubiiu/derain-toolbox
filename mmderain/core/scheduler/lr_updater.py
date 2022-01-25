# This code is taken from https://github.com/open-mmlab/mmediting
# Modified by Raymond Wong

import numbers
from typing import List, Sequence, Tuple

from mmcv.runner import HOOKS, LrUpdaterHook


@HOOKS.register_module()
class LinearLrUpdaterHook(LrUpdaterHook):
    """Linear learning rate scheduler for image generation.

    In the beginning, the learning rate is 'base_lr' defined in mmcv.
    We give a target learning rate 'target_lr' and a start point 'start'
    (iteration / epoch). Before 'start', we fix learning rate as 'base_lr';
    After 'start', we linearly update learning rate to 'target_lr'.

    Args:
        target_lr (float): The target learning rate. Default: 0.
        start (int): The start point (iteration / epoch, specified by args
            'by_epoch' in its parent class in mmcv) to update learning rate.
            Default: 0.
        interval (int): The interval to update the learning rate. Default: 1.
    """

    def __init__(self, target_lr=0, start=0, interval=1, **kwargs):
        super().__init__(**kwargs)
        self.target_lr = target_lr
        self.start = start
        self.interval = interval

    def get_lr(self, runner, base_lr):
        """Calculates the learning rate.

        Args:
            runner (object): The passed runner.
            base_lr (float): Base learning rate.

        Returns:
            float: Current learning rate.
        """
        if self.by_epoch:
            progress = runner.epoch
            max_progress = runner.max_epochs
        else:
            progress = runner.iter
            max_progress = runner.max_iters
        assert max_progress >= self.start

        if max_progress == self.start:
            return base_lr

        # Before 'start', fix lr; After 'start', linearly update lr.
        factor = (max(0, progress - self.start) // self.interval) / (
            (max_progress - self.start) // self.interval)
        return base_lr + (self.target_lr - base_lr) * factor


@HOOKS.register_module()
class PiecewiseLinearLrUpdaterHook(LrUpdaterHook):
    """Piecewise linear parameter scheduler

    This class is adopted from
    https://pytorch.org/ignite/_modules/ignite/handlers/param_scheduler.html#PiecewiseLinear

    Args:
        milestones_values (list[tuple]): list of tuples (event index, parameter value)
            represents milestones and parameter. Milestones should be increasing integers.

    Examples:
        if milestones_values is set to [(1, 1.0), (3, 0.8), (5, 0.5 )], then:
            - Sets lr equal to 1 for till the first iteration
            - Then linearly reduces lr from 1 to 0.8 till the third iteration
            - Then linearly reduces lr from 0.8 to 0.5 till the fifth iteration
    """

    def __init__(self, milestones_values: List[Tuple[int, float]], **kwargs) -> None:
        super().__init__(**kwargs)

        if not isinstance(milestones_values, Sequence):
            raise TypeError(
                f'Argument milestones_values should be a list or tuple, \
                    but given {type(milestones_values)}'
            )
        if len(milestones_values) < 1:
            raise ValueError(
                f'Argument milestones_values should be with at least one value, \
                    but given {milestones_values}'
            )

        values = []  # type: List[float]
        milestones = []  # type: List[int]
        for pair in milestones_values:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    'Argument milestones_values should be a list of pairs (milestone, param_value)')
            if not isinstance(pair[0], numbers.Integral):
                raise TypeError(
                    f'Value of a milestone should be integer, but given {type(pair[0])}')
            if len(milestones) > 0 and pair[0] < milestones[-1]:
                raise ValueError(
                    f'Milestones should be increasing integers, but given {pair[0]} is smaller '
                    f'than the previous milestone {milestones[-1]}'
                )
            milestones.append(pair[0])
            values.append(pair[1])

        self.values = values
        self.milestones = milestones
        self._index = 0

    def _get_start_end(self, event_index) -> Tuple[int, int, float, float]:
        if self.milestones[0] > event_index:
            return event_index - 1, event_index, self.values[0], self.values[0]
        elif self.milestones[-1] <= event_index:
            return (event_index, event_index + 1, self.values[-1], self.values[-1])
        elif self.milestones[self._index] <= event_index < self.milestones[self._index + 1]:
            return (
                self.milestones[self._index],
                self.milestones[self._index + 1],
                self.values[self._index],
                self.values[self._index + 1],
            )
        else:
            self._index += 1
            return self._get_start_end(event_index)

    def get_lr(self, runner, base_lr) -> float:
        event_index = runner.epoch if self.by_epoch else runner.iter
        start_index, end_index, start_value, end_value = self._get_start_end(event_index)
        return start_value + (end_value - start_value) * \
            (event_index - start_index) / (end_index - start_index)


@HOOKS.register_module()
class LadderLrUpdaterHook(LrUpdaterHook):
    """Ladder style parameter scheduler

    Args:
        milestones_values (list[tuple]): list of tuples (event index, parameter value)
            represents milestones and parameter. Milestones should be increasing integers.

    Examples:
        if milestones_values is set to [(1, 1.0), (3, 0.8), (5, 0.5 )], then:
            - Sets lr equal to 1 for till the first iteration
            - Then sets lr to 0.8 till the third iteration
            - Then sets lr to 0.5 till the fifth iteration
    """

    def __init__(self, milestones_values: List[Tuple[int, float]], **kwargs) -> None:
        super().__init__(**kwargs)

        if not isinstance(milestones_values, Sequence):
            raise TypeError(
                f'Argument milestones_values should be a list or tuple, \
                    but given {type(milestones_values)}'
            )
        if len(milestones_values) < 1:
            raise ValueError(
                f'Argument milestones_values should be with at least one value, \
                    but given {milestones_values}'
            )

        values = []  # type: List[float]
        milestones = []  # type: List[int]
        for pair in milestones_values:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    'Argument milestones_values should be a list of pairs (milestone, param_value)')
            if not isinstance(pair[0], numbers.Integral):
                raise TypeError(
                    f'Value of a milestone should be integer, but given {type(pair[0])}')
            if len(milestones) > 0 and pair[0] < milestones[-1]:
                raise ValueError(
                    f'Milestones should be increasing integers, but given {pair[0]} is smaller '
                    f'than the previous milestone {milestones[-1]}'
                )
            milestones.append(pair[0])
            values.append(pair[1])

        self.values = values
        self.milestones = milestones
        self._index = 0

    def _get_lr(self, event_index) -> float:
        if self.milestones[0] > event_index:
            return self.values[0]
        elif self.milestones[-1] <= event_index:
            return self.values[-1]
        elif self.milestones[self._index] <= event_index < self.milestones[self._index + 1]:
            return self.values[self._index + 1]
        else:
            self._index += 1
            return self._get_lr(event_index)

    def get_lr(self, runner, base_lr) -> float:
        event_index = runner.epoch if self.by_epoch else runner.iter
        return self._get_lr(event_index)
