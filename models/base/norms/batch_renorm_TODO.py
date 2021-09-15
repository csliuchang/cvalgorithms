import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm


class BatchRenormalization(_BatchNorm):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True,
                 use_ema_stats=None, update_ema_stats=None,
                 rmax=3, dmax=5, first_step=0,
                 rmax_inc_step=1, dmax_inc_step=1):
        super(BatchRenormalization, self).__init__(
            num_features=num_features,
            eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self._use_ema_stats = use_ema_stats
        self._update_ema_stats = update_ema_stats
        self.rmax = rmax
        self.dmax = dmax
        self.first_step = first_step
        self.rmax_inc_step = rmax_inc_step
        self.dmax_inc_step = dmax_inc_step
        self._rmax_inc_per_step = (rmax-1.0) / rmax_inc_step
        self._dmax_inc_per_step = (dmax-0.0) / dmax_inc_step
        self.step = 0

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    @property
    def use_ema_stats(self) -> bool:
        if self._use_ema_stats is None:
            return self.track_running_stats and not self.training
        else:
            return self._use_ema_stats

    @property
    def update_ema_stats(self) -> bool:
        if self._update_ema_stats is None:
            return self.track_running_stats and self.training
        else:
            return self._update_ema_stats

    @property
    def r_max(self):
        return np.clip(1.0 + self._rmax_inc_per_step * (
            self.step - self.first_step), 1.0, self.rmax)

    @property
    def d_max(self):
        return np.clip(0.0 + self._dmax_inc_per_step * (
            self.step - self.first_step), 0.0, self.dmax)

    def forward(self, input):
        self._check_input_dim(input)

        if self.update_ema_stats:
            self.step += 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.step)
            else:
                exponential_average_factor = np.clip(self.momentum * 1000.0 / float(self.step), 0, 0.1)
        else:
            exponential_average_factor = 0.0


        if False:
            running_mean_np = self.running_mean.cpu().detach().numpy()
            running_var_np = self.running_var.cpu().detach().numpy()
            if len(running_var_np) == 64:
                print('\n---step--- %d' %self.step)
                print('Before F.batch_norm')
                print('self.update_ema_stats', self.update_ema_stats)
                print('self.use_ema_stats', self.use_ema_stats)
                print('exponential_average_factor', exponential_average_factor)
                print(
                    'running_mean',
                    running_mean_np.shape,
                    'min:', np.min(running_mean_np),
                    'max:', np.max(running_mean_np),
                    'mean:', np.mean(running_mean_np))
                print(
                    'running_var',
                    running_var_np.shape,
                    'min:', np.min(running_var_np),
                    'max:', np.max(running_var_np),
                    'mean:', np.mean(running_var_np))

        output = F.batch_norm(
            input,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=not self.use_ema_stats,
            momentum=exponential_average_factor,
            eps=self.eps)

        if False:
            running_mean_np = self.running_mean.cpu().detach().numpy()
            running_var_np = self.running_var.cpu().detach().numpy()
            if len(running_var_np) == 64:
                print('After F.batch_norm')
                print(
                    'running_mean',
                    running_mean_np.shape,
                    'min:', np.min(running_mean_np),
                    'max:', np.max(running_mean_np),
                    'mean:', np.mean(running_mean_np))
                print(
                    'running_var',
                    running_var_np.shape,
                    'min:', np.min(running_var_np),
                    'max:', np.max(running_var_np),
                    'mean:', np.mean(running_var_np))


        if self.training:
            device = self.weight.device
            mean = torch.mean(input, dim=(0,2,3), keepdim=False).to(device)
            std = torch.std(input, dim=(0,2,3), keepdim=False).to(device)


            r = (std / torch.sqrt(self.running_var + self.eps)).to(device)
            r = torch.clamp(r, min=1/self.r_max, max=self.r_max).to(device)
            d = ((mean - self.running_mean) / torch.sqrt(self.running_var + self.eps)).to(device)
            d = torch.clamp(d, min=-self.d_max, max=self.d_max).to(device)


            r = r.view(1, self.num_features, 1, 1)
            d = d.view(1, self.num_features, 1, 1)
            output = output * r + \
                     self.weight.view(1, self.num_features, 1, 1) * d + \
                     self.bias.view(1, self.num_features, 1, 1) * (1 - r)

        return output


if __name__ == '__main__':
    torch.manual_seed(1234)

    inputs = torch.randn([8, 64, 112, 112]).to('cuda:0')

    brn = BatchRenormalization(64)
    brn.train(mode=True)
    brn.cuda(device='cuda:0')

    import time
    st = time.time()

    output = brn(inputs)

    print(time.time()-st, output.size())

    output_np = output.detach().cpu().numpy()
    import numpy as np
    print(np.mean(output_np), np.std(output_np), np.median(output_np))