from typing import Dict


class Average:
    """Implements a simple running average counter."""
    def __init__(self):
        self.total = 0
        self.n_steps = 0

    def update(self, value: float) -> None:
        self.total += value
        self.n_steps += 1

    def compute(self) -> float:
        return self.total / self.n_steps


class DictAverage:
    """Implements a simple running average counter for multiple values."""
    def __init__(self):
        self.average_dict = dict()

    def update(self, values: Dict[str, float]):
        for key, value in values.items():
            if key not in self.average_dict:
                self.average_dict[key] = Average()
            self.average_dict[key].update(value)

    def compute(self) -> Dict[str, float]:
        result = dict()
        for key, average in self.average_dict.items():
            result[key] = average.compute()
        return result
