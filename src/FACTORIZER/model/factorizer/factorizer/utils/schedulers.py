import math

from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
    Decreases learning rate from 1. to 0. over remaining steps.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, last_step=-1):

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_step
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            factor = step / self.warmup_steps
        else:
            progress = step - self.warmup_steps  # progress after warmup
            w = math.pi / (self.total_steps - self.warmup_steps)  # frequency
            factor = 0.5 * (1.0 + math.cos(w * progress))

        return factor
