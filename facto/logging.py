from typing import Dict

from collections import deque


class LossMeter:
    def __init__(self, win_length: int) -> None:
        self.win_length = win_length
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to their initial state."""
        self.steps: int = 0
        self._running: Dict[str, float] = dict()

        self._window: deque[Dict[str, float]] = deque(maxlen=self.win_length)
        self._window_sum: Dict[str, float] = dict()

    def update(self, values: Dict[str, float]) -> None:
        if self.steps == 0:
            for key in values:
                self._running[key] = 0.0
                self._window_sum[key] = 0.0

        assert self._running.keys() == values.keys()

        self.steps += 1

        for key, value in values.items():
            self._running[key] += (value - self._running[key]) / self.steps

        if len(self._window) == self.win_length:
            oldest = self._window[0]
            for key, value in oldest.items():
                self._window_sum[key] -= value

        self._window.append(values)
        for key, value in values.items():
            self._window_sum[key] += value

    @property
    def running_average(self) -> Dict[str, float]:
        return self._running

    @property
    def windowed_average(self) -> Dict[str, float]:
        if len(self._window) == 0:
            return self._window_sum

        win_length = len(self._window)
        return {key: value / win_length for key, value in self._window_sum.items()}

    @property
    def running_totals(self) -> tuple[Dict[str, float], int]:
        totals = {key: value * self.steps for key, value in self._running.items()}
        return totals, self.steps
