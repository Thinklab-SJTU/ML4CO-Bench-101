import numpy as np


class InferenceSchedule(object):
    """
    Map inference step i -> (t1, t2) diffusion timesteps.

    During sampling we only run ``inference_T`` steps, each jumping from
    timestep t1 down to t2 (both in [0, T]). Supported schedules:
      - linear: uniform spacing over [T, 0]
      - cosine: denser early steps via sin schedule
    """

    def __init__(
        self, 
        inference_schedule: str = "linear", 
        T: int = 1000, 
        inference_T: int = 1000
    ):
        """
        Args:
            inference_schedule: "linear" or "cosine"
            T: total training diffusion timesteps
            inference_T: number of sampling steps at inference
        """
        self.inference_schedule = inference_schedule
        self.T = T
        self.inference_T = inference_T

    def __call__(self, i):
        """
        Args:
            i: current inference step index in [0, inference_T)

        Returns:
            t1, t2: start / end diffusion timesteps for this jump
                    (t1 >= t2; progress from large t toward 0)
        """
        assert 0 <= i < self.inference_T

        if self.inference_schedule == "linear":
            # Uniformly map i / (i+1) onto [T, 0]
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t1 = np.clip(t1, 1, self.T)

            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
            t2 = np.clip(t2, 0, self.T - 1)
            return t1, t2
        elif self.inference_schedule == "cosine":
            # sin schedule: smaller jumps near t=T, larger near t=0
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
