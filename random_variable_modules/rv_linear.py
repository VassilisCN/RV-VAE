import torch
import math


class RandomVariableLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(RandomVariableLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = torch.nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    # Initialize weights for RVLinear layer just like PyTorch does
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

    def forward(self, input, distrib_map=None):
        # input must have 1 extra dimension. [b, dist_params=2, *, input_features]
        # for the mean and variance parameters of the distribution
        '''
        Shapes:
            Input: (N,2,*,in_features) N is the batch size, * means any number of additional dimensions
            Weight: (out_features,in_features)
            Bias: (out_features)
            Output: (N,2,*,out_features)
        '''
        # This is done for generic random vars
        input_mean = input[:, 0, ...]
        input_var = input[:, 1, ...]
        # Bias is added only to the mean
        out_mean = torch.nn.functional.linear(input_mean, self.weight, self.bias)
        out_var = torch.nn.functional.linear(input_var, self.weight**2)
        # if out_mean.isnan().any() or out_var.isnan().any():
        #     print('HAS NANS IN: ', self.__class__.__name__)
        out_mean = torch.unsqueeze(out_mean, 1) # At dim = 1 we have our dist_params
        out_var = torch.unsqueeze(out_var, 1)
        return torch.cat((out_mean, out_var), 1)
        
    
if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(1, 2, 3, 3, 1)
    x[:, 1,...] = 1
    x[:, 0,...] = 0
    torch.manual_seed(0)
    l = RandomVariableLinear(1, 2)
    torch.manual_seed(0)
    x1 = x[:,0, ...]
    torch.manual_seed(0)
    l1 = torch.nn.Linear(1, 2)
    y = l(x)
    y1 = l1(x1)
    o = sum(y.view(-1))
    o.backward()
    print(l.weight.grad)
