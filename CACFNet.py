import timm
import torch.nn.functional as F
import torch
from sympy import shape
from torch import nn
# from functools import partial
# from torch.autograd import Variable
# from einops import rearrange
# from timm.models.layers import DropPath
# import cv2
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 groups=1, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=groups),
            norm_layer(out_channels),
            nn.ReLU6()
        )

class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )

# 一个简单的卷积层
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )

class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU6, bias=False, inplace=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
        mask --- [N, T_k]
    output:
        out --- [N, T_q, num_units]
        scores -- [h, N, T_q, T_k]
    '''
    def __init__(self, query_dim: object, key_dim: object, num_heads: object) -> object:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim  # # 键的维度
        self.W_query = nn.Linear(in_features=query_dim, out_features=query_dim, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=key_dim, bias=False)
        self.out = nn.Linear(in_features=key_dim, out_features=key_dim, bias=False)
    def forward(self, query):
        querys = query
        keys = self.W_key(query) + query
        '''
        querys = query  # [N, T_q, num_units]
        keys = self.W_key(query) + query
        '''
        values = self.W_value(query)
        split_size_qk = self.key_dim // self.num_heads
        querys = torch.stack(torch.split(querys, split_size_qk, dim=2), dim=0)
        keys = torch.stack(torch.split(keys, split_size_qk, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size_qk, dim=2), dim=0)
        scores = torch.matmul(querys, keys.transpose(2, 3))
        scores = scores / (split_size_qk ** 0.5)
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)
        out = self.out(out)
        return out

class CrossAttention(nn.Module):

    def __init__(self, query_dim: object, key_dim: object, num_units: object, num_heads: object) -> object:
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.query_expansion = nn.Linear(query_dim, num_units)
        self.W_key = nn.Linear(in_features=key_dim, out_features=key_dim, bias=False)
        self.W_value = nn.Linear(in_features=num_units, out_features=num_units, bias=False)
        self.out = nn.Linear(in_features=num_units, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.query_expansion(query)
        values = self.W_value(key)
        keys=self.W_value(key)
        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        keys = torch.stack(torch.split(keys,split_size, dim=2), dim=0)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)
        scores = torch.matmul(querys, keys.transpose(2, 3))
        scores = scores / (split_size ** 0.5)
        scores = F.softmax(scores, dim=3)
        out = torch.matmul(scores, values)
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)
        out=self.out(out)

        return out

class GLAM(nn.Module):
    def __init__(self, dim, nums_heads=7, weight_ratio=1.0):
        super(GLAM, self).__init__()
        self.out_channels = dim
        self.weight_ratio = weight_ratio
        self.msa_v = MultiHeadAttention(nums_heads, nums_heads,  nums_heads)
        self.cross = CrossAttention(nums_heads, nums_heads, self.out_channels, nums_heads)
        self.local = MFEM(dim, dim)
        self.DWconv=SeparableConvBNReLU(dim,dim,kernel_size=3)

    def forward(self, qk, x):
        n = qk.size(1)
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        qk = F.softmax(qk * self.weight_ratio, dim=1)
        vf = (x).clone()
        vqk_view = qk.clone().permute(0, 3, 2, 1).reshape(b * w, h, -1).contiguous()
        vf_view = vf.permute(0, 3, 2, 1).reshape(b * w, h, c).contiguous()

        x = self.msa_v(vqk_view)
        x_permuted = x.reshape(b, w, h, n).permute(0, 3, 2, 1).contiguous()
        q_view = x_permuted.reshape(b * w, h, -1).contiguous()
        cross = self.cross(q_view, vf_view)
        cross = cross.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()

        x = self.DWconv(vf) + cross
        x = x + self.local(vf)+vf
        x = self.DWconv(x)

        return x

class CGCAM(nn.Module):
    def __init__(self, dim=512, num_heads=6):
        super().__init__()
        self.msa_v = MultiHeadAttention(dim, dim, num_heads)
        self.cross = CrossAttention(dim, dim, dim, num_heads)
        self.DWconv=SeparableConvBNReLU(dim,dim,kernel_size=3)
    def forward(self, x):
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        vf = (x).clone()  # [:, :, :, :].contiguous()  # b,c,h,w
        vqk_view = x.clone().permute(0, 3, 2, 1).reshape(b * w, h, -1).contiguous()
        x = self.msa_v(vqk_view)
        x_permuted = x.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()
        q_view = x_permuted.reshape(b * w, h, -1).contiguous()
        cross = self.cross(q_view, vqk_view)
        cross = cross.reshape(b, w, h, c).permute(0, 3, 2, 1).contiguous()
        x = self.DWconv(vf) + cross
        return x

class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 padding_mode="zeros", with_bn=False, bn_kwargs=None, norm=True):
        super().__init__()
        if bn_kwargs is None:  # 传递给批量归一化层的其他关键字参数，如果没有提供，则默认为None。
            bn_kwargs = {}
        self.add_module("pw", torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        ))
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))
        self.add_module("dw", torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode=padding_mode,
        ))
        if norm:
            self.add_module("active", nn.GELU())

class MFEM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3,5, 7, 9, 11), dilations=(1, 1,1, 1, 1),
                 expansion=1.0, add_identity=True,
                 ):
        super(MFEM, self).__init__()
        self.atten1 = nn.Sequential(
            BSConvU(in_channels * 2, in_channels , 1),
            BSConvU(in_channels , in_channels , 1)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.MEM = MEM(in_channels, out_channels, kernel_sizes, dilations,
                                         expansion=1.0, add_identity=True, )

    def forward(self, x):
        x_avg = self.avgpool(x)
        x_max = self.maxpool(x)
        x_pool = torch.cat((x_avg, x_max), dim=1)
        x_Conv=self.atten1(x_pool)
        attention_weights=self.sigmoid(x_Conv)
        result = self.MEM(x)
        result=result*attention_weights
        return result

class fusion(nn.Module):
    def __init__(self, in_channsel=64, out_channels=64, eps=1e-8):
        super(fusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.Preconv = Conv(in_channels=in_channsel, out_channels=out_channels, kernel_size=1)
        self.post_conv = SeparableConvBNReLU(out_channels, out_channels, 5)

    def forward(self, x, res):
        #x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * self.Preconv(x)
        x = self.post_conv(x)
        return x

class Fusion(nn.Module):
    def __init__(self, in_channsel=64, out_channels=64, eps=1e-8):
        super(Fusion, self).__init__()

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.Preconv = Conv(in_channels=in_channsel, out_channels=out_channels, kernel_size=1)
        self.post_conv = SeparableConvBNReLU(out_channels, out_channels, 5)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU6()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * res + fuse_weights[1] * self.Preconv(x)
        x = self.post_conv(x)
        return x

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        weight, bias, y = weight.contiguous(), bias.contiguous(), y.contiguous()  # avoid cuda error
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        # y, var, weight = ctx.saved_variables
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LN2d(nn.Module):

    def __init__(self, channels, eps=1e-6, requires_grad=True):
        super(LN2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels), requires_grad=requires_grad))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels), requires_grad=requires_grad))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Ti_CA(nn.Module):
    def __init__(self, c, num_heads=8):
        super(Ti_CA, self).__init__()
        #super().__init__()
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.num_heads = num_heads
        self.norm_d = LN2d(c)
        self.norm_g = LN2d(c)
        self.d_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.d_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.g_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.fc_layers = nn.Sequential(
            Conv(c,c // 8, kernel_size=1),  # ratio=8
            nn.ReLU(),
            Conv(c // 8, c, kernel_size=1)
        )

    def forward(self, x_d, x_g):
        b,c,h,w = x_d.shape
        Q_d = self.d_proj1(self.norm_d(x_d))  # B, C, H, W
        Q_g_T = self.g_proj1(self.norm_g(x_g)) # B, C, H, W
        V_d = self.d_proj2(x_d)  # B, C, H, W
        V_g = self.g_proj2(x_g) # B, C, H, W
        Q_d = rearrange(Q_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        Q_g_T = rearrange(Q_g_T, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_d = rearrange(V_d, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        V_g = rearrange(V_g, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        Q_d = torch.nn.functional.normalize(Q_d, dim=-1)
        Q_g_T = torch.nn.functional.normalize(Q_g_T, dim=-1)

        # (B, head, c, hw) x (B, head, hw, c) -> (B, head, c, c)
        attention = (Q_d @ Q_g_T.transpose(-2,-1)) * self.scale

        F_g2d = torch.matmul(torch.softmax(attention, dim=-1), V_g)  # B, head, c, hw
        F_d2g = torch.matmul(torch.softmax(attention, dim=-1), V_d)  # B, head, c, hw
        # scale
        F_g2d = rearrange(F_g2d, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        F_d2g = rearrange(F_d2g, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return x_d + F_g2d * self.beta, x_g + F_d2g * self.gamma

class CFM(nn.Module):
    def __init__(self,dim):
        super(CFM, self).__init__()
        self.attn=Ti_CA(dim)
        self.post_conv = SeparableConvBNReLU(dim, dim, kernel_size=3)
    def  forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x,res=self.attn(x,res)
        x=res+x
        x=self.post_conv(x)
        return  x

class MEM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7, 9, 11),
                 dilations=(1, 1, 1, 1, 1), expansion=1.0, bn_kwargs=None,add_identity=True, ):
        super(MEM, self).__init__()
        out_channels = out_channels
        hidden_channels =in_channels//2
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, 0.001, 0.03),
            nn.GELU(),
            #nn.ReLU(),

        )

        self.dw_convs = nn.ModuleList([

            BSConvU(hidden_channels, hidden_channels, kernel_size=ks, stride=1, padding=ks // 2, dilation=dil,

                    bias=False, with_bn=True, bn_kwargs=bn_kwargs, norm=True)

            for ks, dil in zip(kernel_sizes, dilations)

        ])
        self.pw_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * len(kernel_sizes), hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels, 0.001, 0.03),
            nn.GELU(),
           # nn.ReLU(),
        )
        self.add_identity = add_identity and in_channels == out_channels
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, 0.001, 0.03),
            nn.GELU(),
            #nn.ReLU(),
        )
    def forward(self, x):
        identity = x
        x = self.pre_conv(x)
        branch_outputs = [dw_conv(x) for dw_conv in self.dw_convs]
        x = torch.cat(branch_outputs, dim=1)
        x = self.pw_conv(x)
        x = self.post_conv(x)
        if self.add_identity:
            x=x + identity
        return x

class CFSH(nn.Module):
    def __init__(self, dim, fc_ratio, dilation=[1, 2, 4, 8], dropout=0., num_classes=6):
        super(CFSH, self).__init__()
        self.aspp = MEM(dim, dim, kernel_sizes=(3, 5, 7, 9, 11), dilations=(1, 1, 1, 1, 1),
                                         expansion=1.0, add_identity=True, )
        self.head = nn.Sequential(SeparableConvBNReLU(dim, dim, kernel_size=3),
                                  nn.Dropout2d(p=dropout, inplace=True),
                                  Conv(dim, num_classes, kernel_size=1))
    def forward(self, x):

        aspp=self.aspp(x)
        out = self.head(aspp)

        return out


# 用于从特征图中生成最终的分割结果。
class SegHead(nn.Module):

    def __init__(self, in_channels=64, num_classes=8):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

        self.qkconv_out = nn.Sequential(ConvBNReLU(num_classes, num_classes),
                                        nn.Dropout(0.1),
                                        Conv(num_classes, num_classes, kernel_size=1))

    def forward(self, x, h, w):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        aux = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)

        return feat, aux


class Decoder(nn.Module):
    def __init__(self,
                 encode_channels=[256, 512, 1024, 2048],
                 decode_channels=[256, 512, 1024, 2048],
                 dilation=[[1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
                 fc_ratio=4,
                 dropout=0.1,
                 num_classes=6,
                 weight_ratio=1.0):
        super(Decoder, self).__init__()

        self.Conv1 = ConvBNReLU(encode_channels[-1], decode_channels[-1], 1)
        self.Conv2 = ConvBNReLU(encode_channels[-2], decode_channels[-2], 1)
        self.Conv3 = ConvBNReLU(encode_channels[-3], decode_channels[-3], 1)
        self.Conv4 = ConvBNReLU(encode_channels[-4], decode_channels[-4], 1)

        self.b4 = CGCAM(dim=decode_channels[-1], num_heads=num_classes)

        self.p3 = Fusion(decode_channels[-1], decode_channels[-2])
        self.b3 = GLAM(dim=decode_channels[-2], nums_heads=num_classes,  weight_ratio=weight_ratio)

        self.p2 = Fusion(decode_channels[-2], decode_channels[-3])
        self.b2 = GLAM(dim=decode_channels[-3], nums_heads=num_classes,  weight_ratio=weight_ratio)

        self.p1 = Fusion(decode_channels[-3], decode_channels[-4])
        self.b1 = GLAM(dim=decode_channels[-4], nums_heads=num_classes,  weight_ratio=weight_ratio)

        self.Conv5 = ConvBN(decode_channels[-4], 64, 1)

        self.p = CFM(64)
        self.seg_head = CFSH(64, fc_ratio=fc_ratio, dilation=dilation[3], dropout=dropout, num_classes=num_classes)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.aux_head4 = SegHead(decode_channels[-1], num_classes)
        self.aux_head3 = SegHead(decode_channels[-2], num_classes)
        self.aux_head2 = SegHead(decode_channels[-3], num_classes)
        self.aux_head1 = SegHead(decode_channels[-4], num_classes)
        self.init_weight()

    def forward(self, res, res1, res2, res3, res4, h, w):

        res4 = self.Conv1(res4)
        res3 = self.Conv2(res3)
        res2 = self.Conv3(res2)
        res1 = self.Conv4(res1)

        aux4_4, aux4 = self.aux_head4(res4, h, w)
        x= self.b4(res4)

        x = self.p3(x, res3)
        aux3_3, aux3 = self.aux_head3(x, h, w)
        x= self.b3(aux3_3, x)

        x = self.p2(x, res2)
        aux2_2, aux2 = self.aux_head2(x, h, w)
        x= self.b2(aux2_2, x)

        x = self.p1(x, res1)
        aux1_1, aux1 = self.aux_head1(x, h, w)
        x= self.b1(aux1_1, x)

        x = self.Conv5(x)
        x = self.p(x, res)
        x = self.seg_head(x)

        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x, aux1, aux2, aux3, aux4, aux1_1, aux2_2, aux3_3, aux4_4

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class CGGLNet(nn.Module):
    def __init__(self, num_classes,
                 dropout=0.1,
                 fc_ratio=4,
                 decode_channels=[32, 32, 48,64]):
        super(CGGLNet, self).__init__()
        pretrained_cfg = timm.models.create_model('swsl_resnet50', features_only=True, output_stride=32,
                                                  out_indices=(1, 2, 3, 4)).default_cfg
        pretrained_cfg[
            'file'] = r'E:\Experiment\code\FBSNet-main\resnet50-0676ba61.pth'
        self.backbone = timm.models.swsl_resnet50(pretrained=True, pretrained_cfg=pretrained_cfg)

        encoder_channels = [info['num_chs'] for info in self.backbone.feature_info]

        self.cnn = nn.Sequential(self.backbone.conv1,
                                 self.backbone.bn1,
                                 self.backbone.act1
                                 )
        self.cnn1 = nn.Sequential(self.backbone.maxpool, self.backbone.layer1)
        self.cnn2 = self.backbone.layer2
        self.cnn3 = self.backbone.layer3
        self.cnn4 = self.backbone.layer4

        decode_channels = [decode_channels[-4]* num_classes, decode_channels[-3] * num_classes,
                           decode_channels[-2] * num_classes, decode_channels[-1] * num_classes]

        self.decoder = Decoder(encoder_channels, decode_channels=decode_channels,
                               dropout=dropout, num_classes=num_classes, weight_ratio=1.0)

    def forward(self, x):
        h, w = x.size()[-2:]

        # Encoder ResNet50
        x_pre = self.cnn(x)  ##H/2
        res1 = self.cnn1(x_pre)  ##H/4
        res2 = self.cnn2(res1)  ##H/8
        res3 = self.cnn3(res2)  ##H/16
        res4 = self.cnn4(res3)  ##H/32

        ##
        out, aux1, aux2, aux3, aux4, aux1_1, aux2_2, aux3_3, aux4_4 = self.decoder(x_pre, res1, res2, res3, res4, h, w)

        if self.training:

            return out, aux1, aux2, aux3
        else:

            return out


if __name__ == '__main__':
    num_classes = 6
    in_batch, inchannel, in_h, in_w = 1, 3, 1024, 1024
    x = torch.randn(in_batch, inchannel, in_h, in_w)
    net = CGGLNet(num_classes)
    out, aux2, aux3, aux4 = net(x)
    print(out.shape)
    from thop import profile

    flops, params = profile(net, (x,))  # 注意输入需要是一个元组，即使只有一个元素）下的FLOPs和参数数量
    print('flops: ', flops, 'params: ', params)