import torch
import math
import collections.abc
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)

class RandomVariableConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros'):
        super(RandomVariableConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        def _reverse_repeat_tuple(t, n):
            r"""Reverse the order of `t` and repeat each element for `n` times.
            This can be used to translate padding arg used by Conv and Pooling modules
            to the ones used by `F.pad`.
            """
            return tuple(x for x in reversed(t) for _ in range(n))

        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        self.weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # Initialize weights for RVConv2d layer just like PyTorch does
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def _conv_forward(self, input, weight):
        input_mean = input[:, 0, ...] 
        input_var = input[:, 1, ...]
        if self.padding_mode != 'zeros':
            # We add bias only to the mean of our distributions
            out_mean = torch.nn.functional.conv2d(torch.nn.functional.pad(input_mean, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                                                        weight, self.bias, self.stride, _pair(0), self.dilation, self.groups) 
            out_var = torch.nn.functional.conv2d(torch.nn.functional.pad(input_var, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                                                        weight**2, None, self.stride, _pair(0), self.dilation, self.groups) 
            out_mean = torch.unsqueeze(out_mean, 1) # At dim = 1 we have our dist_params
            out_var = torch.unsqueeze(out_var, 1)
            return torch.cat((out_mean, out_var), 1)

        # We add bias only to the mean of our distributions
        out_mean = torch.nn.functional.conv2d(input_mean, weight, self.bias, self.stride,
                                                self.padding, self.dilation, self.groups)
        out_var = torch.nn.functional.conv2d(input_var, weight**2, None, self.stride,
                                                self.padding, self.dilation, self.groups)
        # PyTorch's conv is based on fft algorithm, thus it might output low negative values. Using abs() is a temporary solution...  
        out_var = torch.abs(out_var)
        
        out_mean = torch.unsqueeze(out_mean, 1) # At dim = 1 we have our dist_params
        out_var = torch.unsqueeze(out_var, 1)
        # if out_mean.isnan().any() or out_var.isnan().any():
        #     print('HAS NANS IN: ', self.__class__.__name__)
        return torch.cat((out_mean, out_var), 1)

    def forward(self, input):
        """
        Input shape: (N, 2, C_in, H, W)
        Output shape: (N, 2, C_out, H_out, W_out)
        Weight shape: (C_out, C_in/groups, kernel_size[0], kernel_size[1])
        Bias shape: (C_out)
        """
        return self._conv_forward(input, self.weight)


if __name__ == "__main__":
    # torch.manual_seed(0)
    x = torch.normal(torch.ones(12,2,2,2)*2., torch.ones(12,2,2,2))
    c = RandomVariableConv2d(3, 7, 2)
    y = c(x)
    o = sum(y.view(-1))
    o.backward()
    print(c.bias.grad)