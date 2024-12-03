import torch
import math
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from mmcv.runner import _load_checkpoint
from mmseg.ops import resize
from mmseg.utils import get_root_logger




def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SqueezeAxialPositionalEmbedding(nn.Module):
    def __init__(self, dim, shape):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn([1, dim, shape]), requires_grad=True)

    def forward(self, x):
        B, C, N = x.shape
        x = x + F.interpolate(self.pos_embed, size=(N), mode='linear', align_corners=False)
        return x


class Sea_Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))
        self.proj_encode_row = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_rowq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_rowk = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.proj_encode_column = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, self.dh, bn_weight_init=0, norm_cfg=norm_cfg))
        self.pos_emb_columnq = SqueezeAxialPositionalEmbedding(nh_kd, 16)
        self.pos_emb_columnk = SqueezeAxialPositionalEmbedding(nh_kd, 16)

        self.dwconv = Conv2d_BN(2 * self.dh, 2 * self.dh, ks=3, stride=1, pad=1, dilation=1,
                                groups=2 * self.dh, norm_cfg=norm_cfg)
        self.act = activation()
        self.pwconv = Conv2d_BN(2 * self.dh, dim, ks=1, norm_cfg=norm_cfg)
        self.sigmoid = h_sigmoid()

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # detail enhance
        qkv = torch.cat([q, k, v], dim=1)
        qkv = self.act(self.dwconv(qkv))
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        ## squeeze row
        qrow = self.pos_emb_rowq(q.mean(-1)).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)
        krow = self.pos_emb_rowk(k.mean(-1)).reshape(B, self.num_heads, -1, H)
        vrow = v.mean(-1).reshape(B, self.num_heads, -1, H).permute(0, 1, 3, 2)

        attn_row = torch.matmul(qrow, krow) * self.scale
        attn_row = attn_row.softmax(dim=-1)
        xx_row = torch.matmul(attn_row, vrow)  # B nH H C
        xx_row = self.proj_encode_row(xx_row.permute(0, 1, 3, 2).reshape(B, self.dh, H, 1))

        ## squeeze column
        qcolumn = self.pos_emb_columnq(q.mean(-2)).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)
        kcolumn = self.pos_emb_columnk(k.mean(-2)).reshape(B, self.num_heads, -1, W)
        vcolumn = v.mean(-2).reshape(B, self.num_heads, -1, W).permute(0, 1, 3, 2)

        attn_column = torch.matmul(qcolumn, kcolumn) * self.scale
        attn_column = attn_column.softmax(dim=-1)
        xx_column = torch.matmul(attn_column, vcolumn)  # B nH W C
        xx_column = self.proj_encode_column(xx_column.permute(0, 1, 3, 2).reshape(B, self.dh, 1, W))

        xx = xx_row.add(xx_column)
        xx = v.add(xx)
        xx = self.proj(xx)
        xx = self.sigmoid(xx) * qkv
        return xx



class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            ks: int,
            stride: int,
            expand_ratio: int,
            activations=None,
            norm_cfg=dict(type='BN', requires_grad=True)
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(Conv2d_BN(inp, hidden_dim, ks=1, norm_cfg=norm_cfg))
            layers.append(activations())
        layers.extend([
            # dw
            Conv2d_BN(hidden_dim, hidden_dim, ks=ks, stride=stride, pad=ks // 2, groups=hidden_dim, norm_cfg=norm_cfg),
            activations(),
            # pw-linear
            Conv2d_BN(hidden_dim, oup, ks=1, norm_cfg=norm_cfg)
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class TokenPyramidModule(nn.Module):
    def __init__(
            self,
            cfgs,
            out_indices,
            inp_channel=16,
            activation=nn.ReLU,
            norm_cfg=dict(type='BN', requires_grad=True),
            width_mult=1.):
        super().__init__()
        self.out_indices = out_indices

        self.stem = nn.Sequential(
            Conv2d_BN(3, inp_channel, 3, 2, 1, norm_cfg=norm_cfg),
            activation()
        )
        self.cfgs = cfgs

        self.layers = []
        for i, (k, t, c, s) in enumerate(cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = t * inp_channel
            exp_size = _make_divisible(exp_size * width_mult, 8)
            layer_name = 'layer{}'.format(i + 1)
            layer = InvertedResidual(inp_channel, output_channel, ks=k, stride=s, expand_ratio=t, norm_cfg=norm_cfg,
                                     activations=activation)
            self.add_module(layer_name, layer)
            inp_channel = output_channel
            self.layers.append(layer_name)

    def forward(self, x):
        outs = []
        x = self.stem(x)
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

class Block(nn.Module):

    def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Sea_Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio,
                                  activation=act_layer, norm_cfg=norm_cfg)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                       norm_cfg=norm_cfg)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1

class BasicLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, key_dim, num_heads,
                 mlp_ratio=4., attn_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_cfg=dict(type='BN2d', requires_grad=True),
                 act_layer=None):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(Block(
                embedding_dim, key_dim=key_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, attn_ratio=attn_ratio,
                drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_cfg=norm_cfg,
                act_layer=act_layer))

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class InjectionMultiSum(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(InjectionMultiSum, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_g):
        '''
        x_g: global features
        x_l: local features
        '''
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)

        out = local_feat * sig_act + global_feat
        return out


class InjectionMultiSumCBR(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        '''
        local_embedding: conv-bn-relu
        global_embedding: conv-bn-relu
        global_act: conv
        '''
        super(InjectionMultiSumCBR, self).__init__()
        self.norm_cfg = norm_cfg

        self.local_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg)
        self.global_embedding = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg)
        self.global_act = ConvModule(inp, oup, kernel_size=1, norm_cfg=None, act_cfg=None)
        self.act = h_sigmoid()

        self.out_channels = oup

    def forward(self, x_l, x_g):
        B, C, H, W = x_l.shape
        local_feat = self.local_embedding(x_l)
        # kernel
        global_act = self.global_act(x_g)
        global_act = F.interpolate(self.act(global_act), size=(H, W), mode='bilinear', align_corners=False)
        # feat_h
        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=(H, W), mode='bilinear', align_corners=False)
        out = local_feat * global_act + global_feat
        return out


class FuseBlockSum(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(FuseBlockSum, self).__init__()
        self.norm_cfg = norm_cfg

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)

        self.out_channels = oup

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        kernel = self.fuse2(x_h)
        feat_h = F.interpolate(kernel, size=(H, W), mode='bilinear', align_corners=False)
        out = inp + feat_h
        return out


class FuseBlockMulti(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int = 1,
            norm_cfg=dict(type='BN', requires_grad=True),
            activations=None,
    ) -> None:
        super(FuseBlockMulti, self).__init__()
        self.stride = stride
        self.norm_cfg = norm_cfg
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out


SIM_BLOCK = {
    "fuse_sum": FuseBlockSum,
    "fuse_multi": FuseBlockMulti,
    "muli_sum": InjectionMultiSum,
    "muli_sum_cbr": InjectionMultiSumCBR,
}


class Topformer(BaseModule):
    def __init__(self, cfgs,
                 channels,
                 out_channels,
                 embed_out_indice,
                 decode_out_indices=[1, 2, 3],
                 depths=3,
                 key_dim=16,
                 num_heads=4,
                 attn_ratios=2,
                 mlp_ratios=2,
                 c2t_stride=2,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_layer=nn.ReLU6,
                 injection_type="muli_sum",
                 init_cfg=None,
                 injection=False):
        super().__init__()
        # print('injection',injection)
        self.channels = channels
        self.norm_cfg = norm_cfg
        self.injection = injection
        self.embed_dim = sum(channels)
        self.decode_out_indices = decode_out_indices
        self.init_cfg = init_cfg
        if self.init_cfg != None:
            self.pretrained = self.init_cfg['checkpoint']
        else:
            self.pretrained = None
        self.tpm = TokenPyramidModule(cfgs=cfgs, out_indices=embed_out_indice, norm_cfg=norm_cfg)
        self.ppa = PyramidPoolAgg(stride=c2t_stride)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule
        self.trans = BasicLayer(
            block_num=3,
            embedding_dim=self.embed_dim,
            key_dim=key_dim,
            num_heads=4,
            mlp_ratio=mlp_ratios,
            attn_ratio=attn_ratios,
            drop=0, attn_drop=0,
            drop_path=dpr,
            norm_cfg=norm_cfg,
            act_layer=act_layer)
        # SemanticInjectionModule
        self.SIM = nn.ModuleList()
        inj_module = SIM_BLOCK[injection_type]
        if self.injection:
            for i in range(len(channels)):
                if i in decode_out_indices:
                    self.SIM.append(
                        inj_module(channels[i], out_channels[i], norm_cfg=norm_cfg, activations=act_layer))
                else:
                    self.SIM.append(nn.Identity())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                n //= m.groups
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        if isinstance(self.pretrained, str):
            print("loading model...")
            logger = get_root_logger()
            checkpoint = _load_checkpoint(self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict_ema' in checkpoint:
                state_dict = checkpoint['state_dict_ema']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)
            print("finished...")

    def forward(self, x):
        ouputs = self.tpm(x)
        out = self.ppa(ouputs)
        out = self.trans(out)
        if self.injection:
            xx = out.split(self.channels, dim=1)
            results = []
            for i in range(len(self.channels)):
                if i in self.decode_out_indices:
                    local_tokens = ouputs[i]
                    global_semantics = xx[i]
                    out_ = self.SIM[i](local_tokens, global_semantics)
                    results.append(out_)
            return results
        else:
            ouputs.append(out)
            return ouputs


class SimpleHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, is_dw=False):
        super(SimpleHead, self).__init__()
        self.input_transform = 'multiple_select'
        self.in_channels = [256, 256, 256]
        self.channels = 256
        self.dropout_ratio = 0.1
        self.conv_seg = nn.Conv2d(self.channels, 2, kernel_size=1)
        self.act_cfg = dict(type='ReLU')
        self.norm_cfg = dict(type='BN', requires_grad=True)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None
        self.in_index = [0, 1, 2]
        embedding_dim = self.channels
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )

    def agg_res(self, preds):
        outs = preds[0]
        for pred in preds[1:]:
            pred = resize(pred, size=outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        x = self.agg_res(xx)
        _c = self.linear_fuse(x)
        x = self.cls_seg(_c)
        return x

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]
        return inputs


model_cfgs = dict(
    cfg=[
        # k,  t,  c, s
        [3, 1, 16, 1],  # 1/2        0.464K  17.461M
        [3, 4, 32, 2],  # 1/4 1      3.44K   64.878M
        [3, 3, 32, 1],  # 4.44K   41.772M
        [5, 3, 64, 2],  # 1/8 3      6.776K  29.146M
        [5, 3, 64, 1],  # 13.16K  30.952M
        [3, 3, 128, 2],  # 1/16 5     16.12K  18.369M
        [3, 3, 128, 1],  # 41.68K  24.508M
        [5, 6, 160, 2],  # 1/32 7     0.129M  36.385M
        [5, 6, 160, 1],  # 0.335M  49.298M
        [3, 6, 160, 1],  # 0.335M  49.298M
    ],
    channels=[32, 64, 128, 160],
    out_channels=[None, 256, 256, 256],
    embed_out_indice=[2, 4, 6, 9],
    decode_out_indices=[1, 2, 3],
    num_heads=8,
    c2t_stride=2,
)


class Seaformernet(nn.Module):
    def __init__(self):
        super(Seaformernet, self).__init__()
        self.seaformer = Topformer(cfgs=model_cfgs.get("cfg"),channels=model_cfgs.get("channels"),
                                   out_channels=[None, 256, 256, 256],
                                   embed_out_indice=model_cfgs.get("embed_out_indice"),injection=True)
        self.seghead = SimpleHead()

    def forward(self, x):
        out = self.seaformer(x)
        out = self.seghead(out)
        return out

if __name__ == '__main__':

    seaformer = Seaformernet()
    a = torch.Tensor(2, 3, 448, 448).cuda()
    seaformer.cuda()
    rst = seaformer(a)
    print(rst.size())


    # topformer = Topformer_regression().cuda()



