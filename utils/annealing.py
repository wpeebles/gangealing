import math
import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts
import warnings


def get_psi_annealing_fn(anneal_fn):
    if anneal_fn == 'linear':
        return linear_anneal
    elif anneal_fn == 'cosine':
        return cosine_anneal
    else:
        raise NotImplementedError


def fastslow_anneal(i, maxval, minval, num_steps, a=0.3):
    """
    This function quickly decays from 1.0, then slowly decays to 0.0. The "a" parameter
    (which should be greater than 0) controls the curvature of the annealing. A large value of
    "a" (e.g., greater than 1) leads to an approximately-linear annealing. A value of
    "a" close to 0 leads to a rapid decay from 1.0 and a very gradual decay to 0.0.
    """
    assert maxval == 1.0
    assert minval == 0.0
    na = num_steps * a
    val = (na - a * i) / (na + i)
    return torch.tensor(val, dtype=torch.float)


def cosine_anneal(i, maxval, minval, num_steps):
    val = minval + 0.5 * (maxval - minval) * (1 + torch.cos(torch.tensor(math.pi * i / num_steps)))
    return val


def linear_anneal(i, maxval, minval, num_steps):
    val = maxval - i * (maxval - minval) / num_steps
    return torch.tensor(val, dtype=torch.float)


def lr_cycle_iters(anneal_psi, period, iter, tm):
    zero_lr_iters = [anneal_psi - 1]
    num_cycles = int(math.log((iter - anneal_psi) / period, tm))
    for n in range(num_cycles):
        step = zero_lr_iters[-1] + period * tm ** n
        zero_lr_iters.append(int(step))
    print(f'Learning Rate Cycles: {zero_lr_iters}')
    return zero_lr_iters


class DecayingCosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, decay=0.9, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay = decay
        self.cur_decay = 1.0

        super(DecayingCosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

        self.T_cur = self.last_epoch

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.cur_decay * (self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        """Step could be called after every batch update"""

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    n = int(epoch // self.T_0)
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                n = 0
        self.cur_decay = self.decay ** n
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


def _test():
    import torch.nn as nn
    import torch
    net = nn.Conv2d(3, 3, 3)
    opt = torch.optim.SGD(net.parameters(), 1.0)
    net2 = nn.Conv2d(3, 3, 3)
    opt2 = torch.optim.SGD(net2.parameters(), 1.0)
    sched1 = DecayingCosineAnnealingWarmRestarts(opt, decay=0.9, T_0=4, T_mult=2)
    sched2 = CosineAnnealingWarmRestarts(opt2, T_0=4, T_mult=2)
    for i in range(20):
        print(sched1.get_last_lr()[0] / sched2.get_last_lr()[0])
        sched1.step(i)
        sched2.step(i)


if __name__ == '__main__':
    _test()
