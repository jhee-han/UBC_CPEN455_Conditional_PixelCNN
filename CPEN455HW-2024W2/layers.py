import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
from utils import *

class nin(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(nin, self).__init__()
        self.lin_a = wn(nn.Linear(dim_in, dim_out))
        self.dim_out = dim_out

    def forward(self, x):
        og_x = x
        # assumes pytorch ordering
        """ a network in network layer (1x1 CONV) """
        # TODO : try with original ordering
        x = x.permute(0, 2, 3, 1)
        shp = [int(y) for y in x.size()]
        out = self.lin_a(x.contiguous().view(shp[0]*shp[1]*shp[2], shp[3]))
        shp[-1] = self.dim_out
        out = out.view(shp)
        return out.permute(0, 3, 1, 2)


class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1),
                    shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad  = nn.ZeroPad2d((int((filter_size[1] - 1) / 2), # pad left
                                  int((filter_size[1] - 1) / 2), # pad right
                                  filter_size[0] - 1,            # pad top
                                  0) )                           # pad down

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down :
            self.down_shift = lambda x : down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,3), stride=(1,1)):
        super(down_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size, stride,
                                            output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        return x[:, :, :(xs[2] - self.filter_size[0] + 1),
                 int((self.filter_size[1] - 1) / 2):(xs[3] - int((self.filter_size[1] - 1) / 2))]


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right :
            self.right_shift = lambda x : right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x


class down_right_shifted_deconv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2,2), stride=(1,1),
                    shift_output_right=False):
        super(down_right_shifted_deconv2d, self).__init__()
        self.deconv = wn(nn.ConvTranspose2d(num_filters_in, num_filters_out, filter_size,
                                                stride, output_padding=1))
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        x = self.deconv(x)
        xs = [int(y) for y in x.size()]
        x = x[:, :, :(xs[2] - self.filter_size[0] + 1):, :(xs[3] - self.filter_size[1] + 1)]
        return x


'''
skip connection parameter : 0 = no skip connection
                            1 = skip connection where skip input size === input size
                            2 = skip connection where skip input size === 2 * input size
'''
class gated_resnet(nn.Module):
    def __init__(self, num_filters, conv_op, nonlinearity=concat_elu, skip_connection=0, film=False,embedding_dim=None,idx=None):
        super(gated_resnet, self).__init__()
        self.skip_connection = skip_connection
        self.nonlinearity = nonlinearity
        self.conv_input = conv_op(2 * num_filters, num_filters) # cuz of concat elu
        self.film = film
        self.film_idx = idx

        if self.film:
            self.film_gamma = nn.Linear(embedding_dim, num_filters)
            self.film_beta  = nn.Linear(embedding_dim, num_filters)

            nn.init.zeros_(self.film_gamma.weight)   # W = 0
            nn.init.ones_ (self.film_gamma.bias)     # b = 1  → gamma ≈ 1
            nn.init.zeros_(self.film_beta.weight)    # W = 0
            nn.init.zeros_(self.film_beta.bias)      # b = 0  → beta  ≈ 0

            nn.init.xavier_uniform_(self.film_gamma.weight)
            nn.init.constant_(self.film_gamma.bias, 1.0)
            nn.init.xavier_uniform_(self.film_beta.weight)
            nn.init.constant_(self.film_beta.bias, 1.0)

            # self.film_gamma = nn.ModuleList([nn.Linear(embedding_dim, num_filters) for _ in range(6)])
            # self.film_beta  = nn.ModuleList([nn.Linear(embedding_dim, num_filters) for _ in range(6)])
            
            # for gamma in self.film_gamma:
            #     nn.init.zeros_(gamma.weight)
            #     nn.init.ones_(gamma.bias)

            # for beta in self.film_beta:
            #     nn.init.zeros_(beta.weight)
            #     nn.init.zeros_(beta.bias)

            # for g,b in zip(self.film_gamma, self.film_beta):
            #     nn.init.zeros_(g.weight); nn.init.ones_(g.bias)
            #     nn.init.zeros_(b.weight); nn.init.zeros_(b.bias)

            # nn.init.ones_(self.film_gamma.bias)
            # nn.init.zeros_(self.film_gamma.weight)



        if skip_connection != 0 :
            self.nin_skip = nin(2 * skip_connection * num_filters, num_filters)

        self.dropout = nn.Dropout2d(0.5)
        self.conv_out = conv_op(2 * num_filters, 2 * num_filters)


    def forward(self, og_x, a=None, class_embed=None):
        x = self.conv_input(self.nonlinearity(og_x))
        if a is not None :
            x += self.nin_skip(self.nonlinearity(a))

        if self.film and class_embed is not None:
            gamma = self.film_gamma(class_embed).unsqueeze(-1).unsqueeze(-1) 
            beta = self.film_beta(class_embed).unsqueeze(-1).unsqueeze(-1)

            # if self.film_idx is None:
            #     raise ValueError("film_idx is None. Ensure it is properly set when creating the gated_resnet instance.")
            # m = self.film_gamma[self.film_idx](class_embed)[:, :, None, None]
            # n = self.film_beta[self.film_idx](class_embed)[:, :, None, None]  # Uncommented this line
            # x = m * x + n

            #debugging_film
            # if self.film and (not hasattr(self, "_dbg_printed")):
            #     with torch.no_grad():
            #         print(f"[FiLM] gamma μ={gamma.mean():.3f}, σ={gamma.std():.3f}")
            x = gamma * x + beta

        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        a, b = torch.chunk(x, 2, dim=1)
        c3 = a * F.sigmoid(b)
        return og_x + c3
