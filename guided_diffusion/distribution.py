from .utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 1

        self.blocks = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            layers = []
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

            self.blocks.append(nn.Sequential(*layers))
            self.blocks[i].apply(init_weights)

        # self.layers = nn.Sequential(*layers)

        # self.layers.apply(init_weights)

    def forward(self, x):
        output = []
        x = x.float()
        for i in range(len(self.blocks)):
            x = self.blocks[i].to('cuda')(x)
            output.append(x)
        return output

# class ConvL(nn.Sequential):
#     def __init__(self, latent_dim, num_filters, i):
#         super(ConvL,self).__init__
#         self.add_module('layer',nn.Conv2d(num_filters[-1], 2 * latent_dim*(len(num_filters)-i), (1,1), stride=1))

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        #self.conv_layer = []
        #for i in range(len(num_filters)):
        #self.conv_layer0=nn.Conv2d(num_filters[-4], 2 * latent_dim*(len(num_filters)-0), (1,1), stride=1)
        """
        self.conv_layer1=nn.Conv2d(num_filters[-3], 2 * latent_dim*(len(num_filters)-1), (1,1), stride=1)
        self.conv_layer2=nn.Conv2d(num_filters[-2], 2 * latent_dim*(len(num_filters)-2), (1,1), stride=1)
        self.conv_layer3=nn.Conv2d(num_filters[-1], 2 * latent_dim*(len(num_filters)-3), (1,1), stride=1)
        """
        self.conv_layer1=nn.Conv2d(num_filters[-3], 2 * latent_dim, (1,1), stride=1)
        self.conv_layer2=nn.Conv2d(num_filters[-2], 2 * latent_dim, (1,1), stride=1)
        self.conv_layer3=nn.Conv2d(num_filters[-1], 2 * latent_dim, (1,1), stride=1)

        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0
        self.resize_down = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        
        # nn.init.kaiming_normal_(self.conv_layer0.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(self.conv_layer0.bias)

        nn.init.kaiming_normal_(self.conv_layer1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer1.bias)

        nn.init.kaiming_normal_(self.conv_layer2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer2.bias)

        nn.init.kaiming_normal_(self.conv_layer3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer3.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        #print(input.shape)
        encoding_all = self.encoder(input)
        #print(encoding.shape)
        self.show_enc = encoding_all
        dists = []
        for i, encoding in enumerate(encoding_all):
            if i == 0:
                continue
            #We only want the mean of the resulting hxw image
            """
            if i > 1:
                feature_down = self.resize_down(dists[i-2].rsample())
                encoding = torch.cat(encoding, feature_down, 1)
            """
            encoding = torch.mean(encoding, dim=2, keepdim=True).double()
            
            encoding = torch.mean(encoding, dim=3, keepdim=True)
            #print(encoding.shape)

            #Convert encoding to 2 x latent dim and split up for mu and log_sigma
            
            #     mu_log_sigma = self.conv_layer0(encoding)
            if i == 1:
                mu_log_sigma = self.conv_layer1(encoding)
            if i == 2:
                mu_log_sigma = self.conv_layer2(encoding)
            if i == 3:
                mu_log_sigma = self.conv_layer3(encoding)
            #print(mu_log_sigma.shape)
            #exit

            #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
            mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

            # mu = mu_log_sigma[:,:self.latent_dim*(len(encoding_all)-i)]
            # log_sigma = mu_log_sigma[:,self.latent_dim*(len(encoding_all)-i):]

            mu = mu_log_sigma[:,:self.latent_dim]
            log_sigma = mu_log_sigma[:,self.latent_dim:]

            #This is a multivariate normal with diagonal covariance matrix sigma
            #https://github.com/pytorch/pytorch/pull/11178
            dists.append(Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1))
        return dists

