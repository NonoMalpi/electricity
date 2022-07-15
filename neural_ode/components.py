import torch

import torch.nn as nn


class NeuralODEfunc(nn.Module):
    def __init__(self, obs_dim: int, hidden_layer_1: int):
        """ Simple neural network for the ODE Solver.

        It consists of two hidden layers with relu activation function

        :param obs_dim: The observable dimension.
        :param hidden_layer_1: Dimension of the first hidden layer
        """
        super(NeuralODEfunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_layer_1),
            nn.Tanh(),
            nn.Linear(hidden_layer_1, obs_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t: torch.Tensor, y: torch.Tensor):
        """ The initial value y is passed through a neural network.

        :param t: Tensor with time to compute the trajectory
        :param y: Tensor with the initial value, shape: (nbatch, obs_dim)
        :return: Tensor with the output of the neural network, shape: (nbatch, obs_dim)
        """
        return self.net(y)


class RecurrentNN(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, nhidden: int, nbatch: int):
        """ Neural network used as encoder in the Variational AutoEncoder

        :param obs_dim: Dimension of the observable space
        :param latent_dim: Dimension of the latent space for the encoder
        :param nhidden: Dimension of the hidden layer
        :param nbatch: Number of samples in batch
        """
        super(RecurrentNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.input_hidden = nn.Linear(obs_dim + nhidden, nhidden)
        # latent_dim is multiplied by 2 for the mean and std of the latent space in VAE
        self.hidden_output = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        """ The input tensor is concatenated with the output of the previous forward pass and
        then passed through a linear layer with nhidden neurons with Tanh as activation function.
        Finally the output is encoded into a latent space with latent_dim * 2 dimension.

        :param x: Tensor containing input to the NN, shape: (nbatch, obs_dim)
        :param h: Tensor containing hidden layer of the previous forward pass, shape: (nbatch, nhidden)
        :return: (Tensor output of the NN, Tensor containing hidden layer after pass)
        """
        combined = torch.cat((x, h), dim=1)  # shape: (nbatch, obs_dim+nhidden)
        h = torch.tanh(self.input_hidden(combined))  # shape: (nbatch, nhidden)
        out = self.hidden_output(h)  # shape: (nbatch, latent_dim * 2)
        return out, h

    def init_hidden(self):
        """ Initialise hidden layer with zeros before first forward pass.

        :return: Tensor containing zeros , shape: (nbatch, nhidden)
        """
        return torch.zeros(self.nbatch, self.nhidden)


class LatentODE(nn.Module):
    def __init__(self, latent_dim: int, nhidden: int):
        """ Neural ODE network to compute the trajectory of the latent space given a initial value
        and a time frame.

        :param latent_dim: Dimension of the latent space
        :param nhidden: Dimension of the hidden layer
        """
        super(LatentODE, self).__init__()
        self.tanh = nn.Tanh()
        self.layer_1 = nn.Linear(latent_dim, nhidden)
        self.layer_2 = nn.Linear(nhidden, nhidden)
        self.layer_3 = nn.Linear(nhidden, latent_dim)

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """ The initial value x is passed through a net consisting in three linear layers with ELU
        activation function.

        :param t: Tensor with time to compute the trajectory
        :param x: Tensor with the initial value, shape: (nbatch, latent_dim)
        :return: Tensor with the output of the neural network, shape: (nbatch, latent_dim)
        """
        out = self.layer_1(x)
        out = self.tanh(out)
        out = self.layer_2(out)
        out = self.tanh(out)
        out = self.layer_3(out)
        return out


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int, nhidden: int):
        """ Neural network that returns computed trajectory in latent space to observable space.

        :param latent_dim: Dimension of the latent space
        :param obs_dim: Dimension of the observable space
        :param nhidden: Dimension of the hidden layer
        """
        super(Decoder, self).__init__()
        self.tanh = nn.Tanh()
        self.layer_1 = nn.Linear(latent_dim, nhidden)
        self.layer_2 = nn.Linear(nhidden, int(nhidden/2))
        self.layer_3 = nn.Linear(int(nhidden/2), obs_dim)

    def forward(self, z: torch.Tensor):
        """ The trajectory in the latent space is decoded through a neural network with two linear
        layers and ReLU activation function.

        :param z: Tensor containing trajectory in latent space, shape: (nbatch, nsamples, latent_dim)
        :return: Tensor containing trajectory in observable space, shape: (nbatch, nsamples, obs_dim)
        """
        out = self.layer_1(z)
        out = self.tanh(out)
        out = self.layer_2(out)
        out = self.tanh(out)
        out = self.layer_3(out)
        return out

