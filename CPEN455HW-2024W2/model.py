import torch.nn as nn
from layers import *
import pdb

class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity,film=False, embedding_dim=None):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=0,film=film, embedding_dim=embedding_dim,idx=i)
                                            for i in range(nr_resnet)])
        #u_stream이라는 instance 생성 
        #Modulelist는 리스트 형태로 여러 레이어 instance를 보관할 때 사용

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1,film=film, embedding_dim=embedding_dim,idx=i)
                                            for i in range(nr_resnet)])

    def forward(self, u, ul,class_embed_vec,class_embed_map):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, class_embed = class_embed_vec)
            ul = self.ul_stream[i](ul, a=u + class_embed_map, class_embed=class_embed_vec) #ul_stream은 gated_resnet의 instance이므로 gasted_resnet의 forward가 실행됨
            u_list  += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity,film=False, embedding_dim=None):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream  = nn.ModuleList([gated_resnet(nr_filters, down_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=1,film=film, embedding_dim=embedding_dim,idx=i)
                                            for i in range(nr_resnet)])

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList([gated_resnet(nr_filters, down_right_shifted_conv2d,
                                        resnet_nonlinearity, skip_connection=2,film=film, embedding_dim=embedding_dim,idx=i)
                                            for i in range(nr_resnet)])

    def forward(self, u, ul, u_list, ul_list,class_embed_vec,class_embed_map):
        for i in range(self.nr_resnet):
            u  = self.u_stream[i](u, a=u_list.pop()+ class_embed_map, class_embed=class_embed_vec)
            a_skip = torch.cat((u, ul_list.pop()), 1)
            # class_embedding_2=class_embedding.size(1) *2
            ul = self.ul_stream[i](ul, a=a_skip, class_embed=class_embed_vec) #(B, embedding_dim)

        return u, ul


class PixelCNN(nn.Module):
    def __init__(self, nr_resnet=5, nr_filters=80, nr_logistic_mix=10,
                    resnet_nonlinearity='concat_elu', input_channels=3, embedding_dim=80,num_classes=4, film=False,fusion_type='add',late_fusion=True, mid_fusion=True, label_dropout_prob=0.1):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == 'concat_elu' :
            self.resnet_nonlinearity = lambda x : concat_elu(x)
        else :
            raise Exception('right now only concat elu is supported as resnet nonlinearity.')

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad  = nn.ZeroPad2d((0, 0, 1, 0))
        self.embedding = nn.Embedding(num_classes, nr_filters)
        self.film = film
        self.embedding_dim = embedding_dim
        self.embedding_dim = nr_filters
        self.late_fusion = late_fusion
        self.fusion_type = fusion_type.lower()

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2 #[5,6,6]
        self.down_layers = nn.ModuleList([PixelCNNLayer_down(down_nr_resnet[i], nr_filters,
                                                self.resnet_nonlinearity,film=self.film, embedding_dim=self.embedding_dim) for i in range(3)])

        self.up_layers   = nn.ModuleList([PixelCNNLayer_up(nr_resnet, nr_filters,
                                                self.resnet_nonlinearity,film=self.film, embedding_dim=self.embedding_dim) for _ in range(3)])

        self.downsize_u_stream  = nn.ModuleList([down_shifted_conv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.downsize_ul_stream = nn.ModuleList([down_right_shifted_conv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.upsize_u_stream  = nn.ModuleList([down_shifted_deconv2d(nr_filters, nr_filters,
                                                    stride=(2,2)) for _ in range(2)])

        self.upsize_ul_stream = nn.ModuleList([down_right_shifted_deconv2d(nr_filters,
                                                    nr_filters, stride=(2,2)) for _ in range(2)])

        self.u_init = down_shifted_conv2d(8, nr_filters, filter_size=(2,3),
                        shift_output_down=True)

        self.ul_init = nn.ModuleList([down_shifted_conv2d(8, nr_filters,
                                            filter_size=(1,3), shift_output_down=True),
                                       down_right_shifted_conv2d(8, nr_filters,
                                            filter_size=(2,1), shift_output_right=True)])

        num_mix = 3 if self.input_channels == 1 else 10

        nin_in_ch = nr_filters if (not late_fusion or fusion_type=='add') else nr_filters + self.embedding_dim
        self.nin_out = nin(nin_in_ch, num_mix * nr_logistic_mix)
        # self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None
        self.label_dropout_prob = label_dropout_prob

        self.mid_fusion = mid_fusion

        if self.mid_fusion:
            self.fuse_u  = nn.Conv2d(nr_filters, nr_filters, 1, bias=False)
            self.fuse_ul = nn.Conv2d(nr_filters, nr_filters, 1, bias=False)


    def forward(self, x,class_labels, sample=False):
        # similar as done in the tf repo :
        class_embed_vec =self.embedding(class_labels)  # (B, embedding_dim)
        # class_embedding = class_embedding.view(class_embedding.size(0),class_embedding.size(1),1,1) # (B, embedding_dim,1,1)

        # #dropout
        # if self.training and self.label_dropout_prob > 0:
        #     dropout_mask = (torch.rand(class_labels.shape[0], device=x.device) < self.label_dropout_prob)
        #     class_embed_vec[dropout_mask] = torch.randn_like(class_embed_vec[dropout_mask]) * 0.01


        class_embed_map = class_embed_vec[:, :, None, None]

        #one hot mask
        class_mask = F.one_hot(class_labels, num_classes=self.embedding.num_embeddings).float()
        class_mask = class_mask[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])  # (B, 4, H, W)

        if self.init_padding is not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample :
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding,class_mask), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding,class_mask), 1)
        u_list  = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)] #초기 feature map생성
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1],class_embed_vec, class_embed_map)
            u_list  += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list  += [self.downsize_u_stream[i](u_list[-1])] #i번째 downsampling layer에 가장 최근 u출력 넣는다.
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u  = u_list.pop()
        ul = ul_list.pop()

        if self.mid_fusion:
            fuse_map = class_embed_map                      # (B,C,1,1)
            u  = u  + self.fuse_u (fuse_map)                # broadcast to spatial
            ul = ul + self.fuse_ul(fuse_map)

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list,class_embed_vec, class_embed_map)

            # upscale (only twice)
            if i != 2 :
                u  = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        if self.late_fusion:
            if self.fusion_type == 'add':
                ul = ul + class_embed_map            # channel‑wise bias
            elif self.fusion_type == 'concat':
                emb_map = class_embed_map.expand(-1, -1, ul.size(2), ul.size(3))
                ul = torch.cat([ul, emb_map], dim=1) # channel concat
            else:
                raise ValueError("fusion_type must be 'add' or 'concat'")

        x_out = self.nin_out(F.elu(ul))
        # pdb.set_trace()

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out
    
    