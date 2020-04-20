# from fvcore
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from time import perf_counter
from typing import Optional


class Timer(object):
    """
    A timer which computes the time elapsed since the start/reset of the timer.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        Reset the timer
        :return:
        """
        self._start = perf_counter()
        self._paused:Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self) -> None:
        """
        Pause the time
        :return:
        """
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused")
        self._paused = perf_counter()

    def is_paused(self) -> bool:
        return self._paused is not None

    def resume(self) -> None:
        """
        Resume the timer
        :return:
        """
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += perf_counter() - self._paused
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        """
        :return: the float number of seconds since the start/reset of
                 the timer, excluding the time when the timer is paused.
        """
        if self._paused is not None:
            end_time: float = self._paused
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

    def avg_seconds(self) -> float:
        """
        :return: the average number of seconds between every start.reset and pause
        """
        return self.seconds() / self._count_start
