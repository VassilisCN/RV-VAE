import torch


class RandomVariableBatchNorm2d(torch.nn.BatchNorm2d):
    '''
        RV implementation of batch normalization 2d RV-BatchNorm2D.
        The implementation is based on the non-RV one found here: https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(RandomVariableBatchNorm2d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
    
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError("expected 5D RV input (got {}D input)".format(input.dim()))

    def forward(self, input):
        """
        Input shape: (N, 2, C, H, W)
        Output shape: (N, 2, C, H, W)
        """
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        # calculate running estimates
        in_mean = input[:, 0, ...] 
        in_var = input[:, 1, ...]
        if self.track_running_stats:
            if self.training:
                n = in_mean.numel() / in_mean.size(1)
                mean = in_mean.mean([0, 2, 3]) # The mean across features is the mean of RV means (mean of mixture of Gaussians)
                var = (in_mean**2+in_var).mean([0, 2, 3]) - mean**2 # and this formula is for variance
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * self.running_mean
                    # update running_var with unbiased var
                    self.running_var = exponential_average_factor * var * n / (n - 1)\
                        + (1 - exponential_average_factor) * self.running_var
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = in_mean.mean([0, 2, 3]) # The mean across features is the mean of RV means (mean of mixture of Gaussians)
            var = (in_mean**2+in_var).mean([0, 2, 3]) - mean**2 # and this formula is for variance

        in_mean = (in_mean - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        in_var = in_var / (var[None, :, None, None] + self.eps)
        if self.affine:
            in_mean = in_mean * self.weight[None, :, None, None] + self.bias[None, :, None, None]
            in_var = in_var * self.weight[None, :, None, None]**2

        mean = torch.unsqueeze(in_mean, 1) # At dim = 1 we have our dist_params
        var = torch.unsqueeze(in_var, 1)
        return torch.cat((mean, var), 1)     