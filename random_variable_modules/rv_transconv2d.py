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

class RandomVariableTransposeConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 output_padding = 0,
                 dilation = 1,
                 groups = 1,
                 bias = True,
                 padding_mode = 'zeros'):
        super(RandomVariableTransposeConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
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
        self.output_padding = output_padding
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
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels // groups, *kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # Initialize weights for RVTransConv2d layer just like PyTorch does
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
    
    def _output_padding(self, input, output_size, stride, padding, kernel_size):

        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        # else:
        #     k = input.dim() - 2
        #     if len(output_size) == k + 2:
        #         output_size = output_size[2:]
        #     if len(output_size) != k:
        #         raise ValueError(
        #             "output_size must have {} or {} elements (got {})"
        #             .format(k, k + 2, len(output_size)))

        #     min_sizes = torch.jit.annotate(List[int], [])
        #     max_sizes = torch.jit.annotate(List[int], [])
        #     for d in range(k):
        #         dim_size = ((input.size(d + 2) - 1) * stride[d] -
        #                     2 * padding[d] + kernel_size[d])
        #         min_sizes.append(dim_size)
        #         max_sizes.append(min_sizes[d] + stride[d] - 1)

        #     for i in range(len(output_size)):
        #         size = output_size[i]
        #         min_size = min_sizes[i]
        #         max_size = max_sizes[i]
        #         if size < min_size or size > max_size:
        #             raise ValueError((
        #                 "requested an output size of {}, but valid sizes range "
        #                 "from {} to {} (for an input of {})").format(
        #                     output_size, min_sizes, max_sizes, input.size()[2:]))

        #     res = torch.jit.annotate(List[int], [])
        #     for d in range(k):
        #         res.append(output_size[d] - min_sizes[d])

        #     ret = res
        return ret

    def forward(self, input):
        """
        Input shape: (N, 2, C_in, H, W)
        Output shape: (N, 2, C_out, H_out, W_out)
        Weight shape: (C_out, C_in/groups, kernel_size[0], kernel_size[1])
        Bias shape: (C_out)
        """
        input_mean = input[:, 0, ...] 
        input_var = input[:, 1, ...]
        output_padding = self._output_padding(input_mean, None, self.stride, self.padding, self.kernel_size)

        # We add bias only to the mean of our distributions
        out_mean = torch.nn.functional.conv_transpose2d(input_mean, self.weight, self.bias, self.stride,
                                                self.padding, output_padding, self.groups, self.dilation)
        out_var = torch.nn.functional.conv_transpose2d(input_var, self.weight**2, None, self.stride,
                                                self.padding, output_padding, self.groups, self.dilation)
        # PyTorch's conv is based on fft algorithm, thus it might output low negative values. Using abs() is a temporary solution...  
        out_var = torch.abs(out_var)
        
        out_mean = torch.unsqueeze(out_mean, 1) # At dim = 1 we have our dist_params
        out_var = torch.unsqueeze(out_var, 1)
        # if out_mean.isnan().any() or out_var.isnan().any():
        #     print('HAS NANS IN: ', self.__class__.__name__)
        return torch.cat((out_mean, out_var), 1)


if __name__ == "__main__":
    # torch.manual_seed(0)
    x = torch.rand(6, 2, 3, 4, 5)
    c = RandomVariableTransposeConv2d(3, 7, 2)
    y = c(x)
    o = sum(y.view(-1))
    o.backward()
    print(c.bias.grad)