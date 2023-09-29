from torchsummary import summary
from random_variable_modules import *
import torch
from torch import nn
from torch.nn import functional as F

"""
Implementation based on: https://github.com/AntixK/PyTorch-VAE
"""
class VanillaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim=128, rv=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.rv = rv
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_out_channels = in_channels

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []
        if self.rv:
            self.decoder_input = RandomVariableLinear(latent_dim, hidden_dims[-1] * 4)
        else:
            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if self.rv:
                modules.append(
                    nn.Sequential(
                        RandomVariableTransposeConv2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        RandomVariableBatchNorm2d(hidden_dims[i + 1]),
                        RandomVariableReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU())
                )



        self.decoder = nn.Sequential(*modules)
        if self.rv:
            self.final_layer = nn.Sequential(
                                RandomVariableTransposeConv2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                                RandomVariableBatchNorm2d(hidden_dims[-1]),
                                RandomVariableReLU(),
                                # nn.LeakyReLU(),
                                RandomVariableConv2d(hidden_dims[-1], out_channels= in_out_channels,
                                        kernel_size= 3, padding= 1),
                                # nn.Sigmoid())
                                nn.Tanh())
        else:
            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                nn.ReLU(),
                                # nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= in_out_channels,
                                        kernel_size= 3, padding= 1),
                                # nn.Sigmoid())
                                nn.Tanh())
        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        # print(torch.mean(mu), torch.mean(log_var))
        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        if self.rv:
            result = result.view(-1, 2, 512, 2, 2)
        else:
            result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # eps = torch.ones_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        if self.rv:
            mu, log_var = self.encode(input) #iason's idea
            # log_var = torch.zeros(mu.shape, device=mu.device) #iason's idea
            out_mean = torch.unsqueeze(mu, 1) # At dim = 1 we have our dist_params
            var      = torch.exp(log_var)
            out_var  = torch.unsqueeze(var, 1)
            z = torch.cat((out_mean, out_var), 1)
            if out_mean.isnan().any() or out_var.isnan().any():
                print(torch.mean(out_mean), torch.mean(out_var))
                print('HAS NANS IN: ', self.__class__.__name__)
                raise ValueError
        else:
            mu, log_var = self.encode(input)
            z = self.reparameterize(mu, log_var)
        return  self.decode(z), mu, log_var


class BetaTCVAE(nn.Module):
    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_dims = None,
                 anneal_steps = 200,
                 alpha = 1.,
                 beta =  6.,
                 gamma = 1.,
                 rv = False,
                 **kwargs):
        super(BetaTCVAE, self).__init__()

        self.latent_dim = latent_dim
        self.anneal_steps = anneal_steps
        self.rv = rv
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 32, 32, 32]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 4, stride= 2, padding  = 1),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        self.fc = nn.Linear(hidden_dims[-1]*16, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)


        # Build Decoder
        modules = []

        if self.rv:
            self.decoder_input = RandomVariableLinear(latent_dim, 256 *  2)
        else:
            self.decoder_input = nn.Linear(latent_dim, 256 *  2)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if self.rv:
                modules.append(
                    nn.Sequential(
                        RandomVariableTransposeConv2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        RandomVariableReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        nn.LeakyReLU())
                )

        self.decoder = nn.Sequential(*modules)

        if self.rv:
            self.final_layer = nn.Sequential(
                                RandomVariableTransposeConv2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                                RandomVariableReLU(),
                                RandomVariableConv2d(hidden_dims[-1], out_channels= 3,
                                        kernel_size= 3, padding= 1),
                                nn.Tanh())

        else:
            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                        kernel_size= 3, padding= 1),
                                nn.Tanh())

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)

        result = torch.flatten(result, start_dim=1)
        result = self.fc(result)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        if self.rv:
            result = result.view(-1, 2, 32, 4, 4)
        else:
            result = result.view(-1, 32, 4, 4)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        if self.rv:
            mu, _ = self.encode(input) #iason's idea
            log_var = torch.zeros(mu.shape, device=mu.device) #iason's idea
            out_mean = torch.unsqueeze(mu, 1) # At dim = 1 we have our dist_params
            var      = torch.exp(0.5*log_var)
            out_var  = torch.unsqueeze(var, 1)
            z = torch.cat((out_mean, out_var), 1)
        else:
            mu, log_var = self.encode(input)
            z = self.reparameterize(mu, log_var)
        return  self.decode(z), mu, log_var, z


class WAE_MMD(nn.Module):
    def __init__(self, in_channels, latent_dim=128, rv=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.rv = rv
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    # nn.ReLU())
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        #TODO: Should I change here something for the RV case?
        self.fc_z = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []
        if self.rv:
            self.decoder_input = RandomVariableLinear(latent_dim, hidden_dims[-1] * 4)
        else:
            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            if self.rv:
                modules.append(
                    nn.Sequential(
                        RandomVariableTransposeConv2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        RandomVariableBatchNorm2d(hidden_dims[i + 1]),
                        RandomVariableReLU())
                        # nn.LeakyReLU())
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        # nn.ReLU())
                        nn.LeakyReLU())
                )



        self.decoder = nn.Sequential(*modules)
        if self.rv:
            self.final_layer = nn.Sequential(
                                RandomVariableTransposeConv2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                                RandomVariableBatchNorm2d(hidden_dims[-1]),
                                RandomVariableReLU(),
                                # nn.LeakyReLU(),
                                RandomVariableConv2d(hidden_dims[-1], out_channels= 3,
                                        kernel_size= 3, padding= 1),
                                # nn.Sigmoid())
                                nn.Tanh())
        else:
            self.final_layer = nn.Sequential(
                                nn.ConvTranspose2d(hidden_dims[-1],
                                                hidden_dims[-1],
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                output_padding=1),
                                nn.BatchNorm2d(hidden_dims[-1]),
                                # nn.ReLU(),
                                nn.LeakyReLU(),
                                nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                        kernel_size= 3, padding= 1),
                                # nn.Sigmoid())
                                nn.Tanh())
        
    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        z = self.fc_z(result)
        return z

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        if self.rv:
            result = result.view(-1, 2, 512, 2, 2)
        else:
            result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, input, **kwargs):
        z = self.encode(input)
        return  self.decode(z), z

if __name__ == "__main__":
    model = VanillaVAE(in_channels=3, latent_dim=128, rv=False)
    x = torch.rand((6,3,64,64))#.to(device)
    y = model(x)