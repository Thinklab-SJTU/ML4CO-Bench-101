import numpy as np


class InferenceSchedule(object):
    def __init__(
        self, inference_schedule: str = "linear", T: int = 1000, inference_T: int = 1000
    ):
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i):
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(
                np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T
            )
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int(
                np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T
            )
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        else:
            raise ValueError(
                "Unknown inference schedule: {}".format(self.inference_schedule)
            )