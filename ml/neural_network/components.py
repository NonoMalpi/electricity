from typing import List, NoReturn

import torch

import torch.nn as nn


class NeuralNetFunc(nn.Module):
    """ Class containing a neural network module used in the neural ODE framework.

    Parameters
    ----------
    obs_dim: int
        The observable dimension of the input and output layers.

    hidden_layer_neurons:
        List containing number of neurons for each hidden layer.

    activation_functions: List[nn.modules.activation.Module]
        List containing the activation function to apply after each layer.

    Attributes
    ----------
    net: torch.nn.Sequential
        The neural net architecture.

    Methods
    -------
    forward
    """
    def __init__(self,
                 obs_dim: int,
                 hidden_layer_neurons: List[int],
                 activation_functions: List[nn.modules.activation.Module]):
        super(NeuralNetFunc, self).__init__()

        self.net = self._build_net_architecture(obs_dim=obs_dim,
                                                hidden_layer_neurons=hidden_layer_neurons,
                                                activation_functions=activation_functions)

        self._initialize_parameters()

    def _build_net_architecture(self,
                                obs_dim: int,
                                hidden_layer_neurons: List[int],
                                activation_functions: List[torch.nn.modules.activation.Module]) -> torch.nn.Sequential:
        """ Generate the sequential neural network achitecture.

        Parameters
        ----------
        obs_dim: int
            The observable dimension of the input and output layers.

        hidden_layer_neurons:
            List containing number of neurons for each hidden layer.

        activation_functions: List[nn.modules.activation.Module]
            List containing the activation function to apply after each layer.

        Returns
        -------
        torch.nn.Sequential
            The sequence of modules constituting the neural network.
        """

        assert len(hidden_layer_neurons) == len(activation_functions), \
            "hidden_layer_neurons list and activation_functions list must be of the same length."

        modules_list = []

        for k in range(len(hidden_layer_neurons)):
            in_features = obs_dim if k == 0 else hidden_layer_neurons[k - 1]
            out_features = hidden_layer_neurons[k]
            modules_list.append(nn.Linear(in_features=in_features, out_features=out_features))
            modules_list.append(activation_functions[k])

        modules_list.append(nn.Linear(in_features=hidden_layer_neurons[-1], out_features=obs_dim))

        return nn.Sequential(*modules_list)

    def _initialize_parameters(self) -> NoReturn:
        """ Initialize weights and bias of linear modules. """
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t: torch.Tensor, y: torch.Tensor):
        """ The forward pass of the neural network in the context of neural ODE.

        Parameters
        ----------
        t: torch.Tensor
            Tensor with time steps to compute the trajectory.

        y: torch.Tensor
            Tensor with the initial value (batch_size, 1, obs_dim).

        Returns
        -------
        torch.Tensor
        Tensor with the output of the neural network (batch_size, obs_dim).
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

