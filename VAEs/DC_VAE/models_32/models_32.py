import torch
from torch import nn
import torch.nn.functional as F
import random_variable_modules as rvm

UP_MODES = ['nearest', 'bilinear']
NORMS = ['in', 'bn']

gf_dim = 256
df_dim = 128
g_spectral_norm = False
d_spectral_norm = True
bottom_width = 4

class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, ksize=3, num_skip_in=0, short_cut=False, norm=None, rv=False):
        super(Cell, self).__init__()
        self.rv = rv
        if rv:
            self.c1 = rvm.RandomVariableConv2d(in_channels, out_channels, ksize, padding=ksize//2)
            self.c2 = rvm.RandomVariableConv2d(out_channels, out_channels, ksize, padding=ksize//2)
        else:
            self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize//2)
            self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=ksize//2)
        assert up_mode in UP_MODES
        self.up_mode = up_mode
        self.norm = norm
        if norm:
            assert norm in NORMS
            if norm == 'bn':
                if rv:
                    self.n1 = rvm.RandomVariableBatchNorm2d(in_channels)
                    self.n2 = rvm.RandomVariableBatchNorm2d(out_channels)
                else:
                    self.n1 = nn.BatchNorm2d(in_channels)
                    self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                if rv:
                    raise NotImplementedError(norm)
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)

        # inner shortcut
        self.c_sc = None
        if short_cut:
            if rv:
                self.c_sc = rvm.RandomVariableConv2d(in_channels, out_channels, kernel_size=1)
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            if rv:
                self.skip_in_ops = nn.ModuleList([rvm.RandomVariableConv2d(out_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])
            else:
                self.skip_in_ops = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def forward(self, x, skip_ft=None):
        residual = x

        # first conv
        if self.norm:
            residual = self.n1(residual)
        if self.rv:
            h = rvm.RandomVariableReLU()(residual)
            h = rvm_interpolate(h, scale_factor=2, mode=self.up_mode)
            _, _, _, ht, wt = h.size()
        else:
            h = nn.ReLU()(residual)
            h = F.interpolate(h, scale_factor=2, mode=self.up_mode)
            _, _, ht, wt = h.size()
        h = self.c1(h)
        h_skip_out = h

        # second conv
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                if self.rv:
                    h += skip_in_op(rvm_interpolate(ft, size=(ht, wt), mode=self.up_mode))
                else:
                    h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode))
        if self.norm:
            h = self.n2(h)
        if self.rv:
            h = rvm.RandomVariableReLU()(h)
        else:
            h = nn.ReLU()(h)
        final_out = self.c2(h)

        # shortcut
        if self.c_sc:
            if self.rv:
                final_out += self.c_sc(rvm_interpolate(x, scale_factor=2, mode=self.up_mode))
            else:
                final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.up_mode))

        return h_skip_out, final_out

def rvm_interpolate(input, size = None, scale_factor = None, mode = 'nearest', align_corners = None):
    means = F.interpolate(input[:,0,...], size, scale_factor, mode, align_corners)
    varis = F.interpolate(input[:,1,...], size, scale_factor, mode, align_corners)
    return torch.cat((torch.unsqueeze(means, 1), torch.unsqueeze(varis, 1)), 1)

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

class OptimizedDisBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class DisBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class Decoder(nn.Module):
    def __init__(self, zdim, rv=False):
        super(Decoder, self).__init__()
        self.zdim = zdim
        self.rv = rv
        self.ch = gf_dim
        self.bottom_width = bottom_width
        if rv:
            self.l1 = rvm.RandomVariableLinear(zdim, (bottom_width ** 2) * gf_dim)
        else:
            self.l1 = nn.Linear(zdim, (bottom_width ** 2) * gf_dim)
        self.cell1 = Cell(gf_dim, gf_dim, 'nearest', num_skip_in=0, short_cut=True, rv=rv)
        self.cell2 = Cell(gf_dim, gf_dim, 'bilinear', num_skip_in=1, short_cut=True, rv=rv)
        self.cell3 = Cell(gf_dim, gf_dim, 'nearest', num_skip_in=2, short_cut=False, rv=rv)
        if rv:
            self.to_rgb = nn.Sequential(
                rvm.RandomVariableBatchNorm2d(gf_dim),
                rvm.RandomVariableReLU(),
                rvm.RandomVariableConv2d(gf_dim, 3, 3, 1, 1),
                nn.Tanh()
            )
        else:
            self.to_rgb = nn.Sequential(
                nn.BatchNorm2d(gf_dim),
                nn.ReLU(),
                nn.Conv2d(gf_dim, 3, 3, 1, 1),
                nn.Tanh()
            )

    def forward(self, z):
        if self.rv:
            h = self.l1(z).view(-1, 2, self.ch, self.bottom_width, self.bottom_width)
        else:
            h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out, ))
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)
        return output


class DualEncoder(nn.Module):
    def __init__(self, cont_dim=16, activation=nn.ReLU()):
        super(DualEncoder, self).__init__()
        self.ch = df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(3, self.ch)
        self.block2 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.head_disc = nn.utils.spectral_norm(nn.Linear(cont_dim, 1))
        self.l5 = nn.Linear(self.ch, cont_dim, bias=False)
        self.l5 = nn.utils.spectral_norm(self.l5)
        self.head_b1 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear( 256, cont_dim, bias=False)
        )
        self.head_b2 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear( 64, cont_dim, bias=False)
        )
        self.head_b3 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear( 64, cont_dim, bias=False)
        )
        self.head_b4 = nn.Sequential(
            nn.Conv2d(self.ch, 1, kernel_size=1, padding=0),
            nn.Flatten(),
            nn.Linear( 64, cont_dim, bias=False)
        )
        
    def forward(self, x, mode="dual"):
        h = x
        h1 = self.block1(h)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        h4 = self.block4(h3)
        h = self.activation(h4)
        h = h.sum(2).sum(2)
        h = self.l5(h)
        disc_out = self.head_disc(h)
        if mode == "dis":
            return disc_out
        elif mode == "cont":
            cont_out = {
                "b1-raw" : h1,
                "b2-raw" : h2,
                "b3-raw" : h3,
                "b4-raw" : h4,
                "b1"     : self.head_b1(h1),
                "b2"     : self.head_b2(h2),
                "b3"     : self.head_b3(h3),
                "b4"     : self.head_b4(h4),
                "final"  : h
            }
            return cont_out
        elif mode=="cont_local":
            cont_out = {
                "local_h1"    : h1, # 128x16x16
                "local_h2"    : h2, # 128x8x8
                "local_h3"    : h3, # 128x8x8
                "local_h4"    : h4, # 128x8x8
                "b1"          : self.head_b1(h1),
                "final" : h
            }
            return cont_out
        # return disc_out, cont_out

class Encoder(nn.Module):
    def __init__(self, zdim, activation=nn.ReLU()):
        super(Encoder, self).__init__()
        self.ch = df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(3, self.ch)
        self.block2 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock(
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.l5 = nn.Linear(self.ch, zdim*2, bias=False)
        self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)
        return output

