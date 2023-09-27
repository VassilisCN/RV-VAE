import torch


class RandomVariableUpsample(torch.nn.Module):
    def __init__(self, size = None, scale_factor = None,
                 mode = 'nearest', align_corners = None):
        super(RandomVariableUpsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        means = torch.nn.functional.interpolate(input[:,0,...], self.size, self.scale_factor, self.mode, self.align_corners)
        varis = torch.nn.functional.interpolate(input[:,1,...], self.size, self.scale_factor, self.mode, self.align_corners)
        return torch.cat((torch.unsqueeze(means, 1), torch.unsqueeze(varis, 1)), 1)

    def extra_repr(self) -> str:
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info
    
if __name__ == "__main__":
    m = torch.zeros((3,1,3,4,4))
    s = torch.ones((3,1,3,4,4))
    ms = torch.cat((m,s),1)
    u = RandomVariableUpsample(scale_factor=2, mode='nearest')
    ums = u(ms)
    print(ums.shape)
