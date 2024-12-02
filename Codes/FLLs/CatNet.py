import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.point_wise_conv = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.point_wise_conv(x)
        return x

class _DenseLayer(nn.Module):
    def __init__(self, inplace, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.ModuleList([
            nn.BatchNorm2d(inplace),
            nn.GELU(),
            SeparableConv(in_channels=inplace, out_channels=bn_size * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.GELU(),
            SeparableConv(in_channels=bn_size * growth_rate,
                      out_channels=growth_rate, kernel_size=3,
                      padding=1, bias=False),
        ])
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        skip_x = x
        for blk in self.dense_layer:
            x = blk(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        return torch.cat([x, skip_x], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, inplances, growth_rate, bn_size, drop_rate=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(_DenseLayer(inplances + i * growth_rate, growth_rate, bn_size, drop_rate))

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x


class _CBAMLayer(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.squeeze_avg = nn.AdaptiveAvgPool2d(1)
        self.squeeze_max = nn.AdaptiveMaxPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=int(channel // ratio)),
            nn.GELU(),
            nn.Linear(in_features=int(channel // ratio), out_features=channel),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.conv = SeparableConv(in_channels=2, out_channels=1, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        y_avg = self.squeeze_avg(x).contiguous().view(b, c)
        y_max = self.squeeze_max(x).contiguous().view(b, c)
        z_avg = self.excitation(y_avg).contiguous().view(b, c, 1, 1)
        z_max = self.excitation(y_max).contiguous().view(b, c, 1, 1)
        z = z_avg + z_max
        z = self.sigmoid(z)
        w = x * z.expand_as(x)
        s_avg = torch.mean(w, dim=1, keepdim=True)
        s_max, _ = torch.max(w, dim=1, keepdim=True)
        s = torch.cat((s_avg, s_max), dim=1)
        s = self.conv(s)
        s = self.sigmoid(s)
        out = w * s.expand_as(w)
        return out


class DenseCBAMBlock(nn.Module):
    def __init__(self, num_layers, inplances, channel, growth_rate, bn_size, drop_rate=0., ratio=16):
        super().__init__()
        self.dense_layer = DenseBlock(num_layers, inplances, growth_rate, bn_size, drop_rate)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(inplances + num_layers * growth_rate),
            nn.GELU(),
            SeparableConv(inplances + num_layers * growth_rate, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )
        self.cbam_layer = _CBAMLayer(channel, ratio)

    def forward(self, x):
        x = self.dense_layer(x)
        x = self.conv(x)
        x = self.cbam_layer(x)
        return x

class CNN_Encoder(nn.Module):
    def __init__(self, encoder_depth, init_channel, channels, num_layers, growth_rate, drop_rate=0.):
        super().__init__()
        self.init_conv = SeparableConv(init_channel, channels, kernel_size=1, bias=False)
        self.layers = nn.ModuleList([])
        for i in range(encoder_depth):
            layer = DenseCBAMBlock(num_layers=num_layers[i], inplances=channels * (2 ** i),
                                   channel=channels * (2 ** (i + 1)), growth_rate=growth_rate[i],
                                   bn_size=4, drop_rate=drop_rate)
            self.layers.append(layer)
            if i < encoder_depth - 1:
                down_sample = SeparableConv(channels * (2 ** (i + 1)),channels * (2 ** (i + 1)),kernel_size=2,
                                        stride=2, bias=False)
                self.layers.append(down_sample)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, gain=1.0)


    def forward(self, x):
        res = []
        x = self.init_conv(x)
        for blk in self.layers:
            x = blk(x)
            if blk._get_name() == "DenseCBAMBlock":
                res.append(x)
        return res

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim # Number of input channels
        self.window_size = window_size
        self.num_heads = num_heads # number of attention heads
        head_dim = dim // num_heads # number of single attention channels
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)) #2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        :param x: input features with shape of (num_windows*B, N, C)
        :param mask:(0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        :return:
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3 ,1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        a = k.transpose(-2, -1)
        attn =q @ a

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_rate=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, H_x, W_x, C = x.shape
        assert H_x == H and W_x == W, "input feature has wrong size"

        x = x.view(B, -1, C)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.view(B, H, W, C)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        H, W = self.input_resolution
        B, H_x, W_x, C = x.shape
        assert H_x == H and W_x == W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = x.view(B, H // 2, W // 2, 2 * C)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
                x = blk(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_chans=3, embed_dim=64):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            SeparableConv(in_chans, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            SeparableConv(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            SeparableConv(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class SwinTransformerEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_chans=3,
                 embed_dim=48, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 从BasicLayer类中生成了一个实例
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
            if i_layer < (self.num_layers - 1):
                down_sample = PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                             patches_resolution[1] // (2 ** i_layer)),
                                           dim=int(embed_dim * 2 ** i_layer),
                                           norm_layer=norm_layer)
                self.layers.append(down_sample)

        self.norm = norm_layer(self.num_features)

    def forward_features(self, x):
        res = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
            if layer._get_name() == "BasicLayer":
                res.append(x.permute(0, 3, 1, 2))

        return res

    def forward(self, x):
        out = self.forward_features(x)
        return out


class FeatureFuse(nn.Module):
    def __init__(self, input_channels, vit_channels, cnn_channels, out_channels):
        super().__init__()
        if input_channels is not None:
            self.signal = 0
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    input_channels + vit_channels,
                    input_channels + vit_channels,
                    kernel_size=2, stride=2),
                nn.BatchNorm2d(input_channels + vit_channels),
                nn.GELU()
            )
            self.channel_fuse = nn.Sequential(
                SeparableConv(in_channels=input_channels + vit_channels + cnn_channels,
                          out_channels=out_channels,
                          kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )
        else:
            self.signal = 1
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=vit_channels,
                    out_channels=vit_channels,
                    kernel_size=2,
                    stride=2),
                nn.BatchNorm2d(vit_channels),
                nn.GELU())
            self.channel_fuse = nn.Sequential(
                SeparableConv(in_channels=vit_channels + cnn_channels,
                          out_channels=out_channels,
                          kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU())

    # signal=1代表最底层的特征融合模块，只有CNN和ViT两个输入
    def forward(self, input_x, vit_x, cnn_x):
        if self.signal == 1:
            x = self.upsample(vit_x)
            x = torch.cat([x, cnn_x], dim=1)
            x = self.channel_fuse(x)
        else:
            x = torch.cat([input_x, vit_x], dim=1)
            x = self.upsample(x)
            x = torch.cat([x, cnn_x], dim=1)
            x = self.channel_fuse(x)
        return x


class SEModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_features=channel, out_features=int(channel // ratio)),
            nn.GELU(),
            nn.Linear(in_features=int(channel // ratio), out_features=channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        x = x * z.expand_as(x)
        return x


def Conv3X3BNGELU(in_channels, out_channels):
    return nn.Sequential(SeparableConv(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1),
                         nn.BatchNorm2d(out_channels),
                         nn.GELU(),
                         SeparableConv(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1),
                         nn.BatchNorm2d(out_channels)
                         )


class ConvPipeline(nn.Module):
    def __init__(self, in_channels, out_channels, attn_ratio=16):
        super().__init__()
        self.conv_layer = Conv3X3BNGELU(in_channels, out_channels)
        self.se_layer = SEModule(out_channels, attn_ratio)
        self.gelu = nn.GELU()

    def forward(self, x):
        res_x = x
        x = self.conv_layer(x)
        x = res_x + x
        x = self.gelu(x)
        x = self.se_layer(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, input_channels, vit_channels, cnn_channels, out_channels):
        super().__init__()
        self.feature_fuse = FeatureFuse(input_channels,
                                        vit_channels,
                                        cnn_channels,
                                        out_channels)
        self.channel_pipeline = ConvPipeline(out_channels, out_channels)

    def forward(self, input_x, vit_x, cnn_x):
        x = self.feature_fuse(input_x, vit_x, cnn_x)
        x = self.channel_pipeline(x)
        return x


class Decoder(nn.Module):
    def __init__(self, depths, num_classes, input_channels, vit_channels, cnn_channels, out_channels):
        super().__init__()
        self.decoder_layers = nn.ModuleList([])
        self.depths = depths
        for i in range(self.depths):
            if i == 0:
                layer = DecoderLayer(
                    input_channels=None,
                    cnn_channels=cnn_channels * (2 ** (self.depths - (i + 1))),
                    vit_channels=vit_channels * (2 ** (self.depths - (i + 1))),
                    out_channels=out_channels * (2 ** (self.depths - (i + 1))))
            else:
                layer = DecoderLayer(
                    input_channels=input_channels * (2 ** (self.depths - (i))),
                    cnn_channels=cnn_channels * (2 ** (self.depths - (i + 1))),
                    vit_channels=vit_channels * (2 ** (self.depths - (i + 1))),
                    out_channels=out_channels * (2 ** (self.depths - (i + 1))))
            self.decoder_layers.append(layer)
        self.classifier = SeparableConv(out_channels, num_classes,kernel_size=1, bias=True)

    def forward(self, vit_features, cnn_features):
        res = []
        for i in range(self.depths):
            while i == 0:
                x = self.decoder_layers[i](None,
                                           vit_features[-1 * i + (-1)],
                                           cnn_features[-1 * i + (-1)])
                res.append(x)
                break
            else:
                x = self.decoder_layers[i](x,
                                           vit_features[-1 * i + (-1)],
                                           cnn_features[-1 * i + (-1)])
                res.append(x)

        x = self.classifier(x)
        return x, res


class CaT_Net_with_Decoder_DeepSup(nn.Module):
    def __init__(self, num_classes, cnn_channels=32, swin_trans_channels=24, num_layers=[4, 4, 4, 4, 4]):
        super().__init__()
        self.name = getattr(CaT_Net_with_Decoder_DeepSup, "__name__")
        self.cnn_encoder = CNN_Encoder(encoder_depth=4,
                                       init_channel=3,
                                       channels=cnn_channels // 2,
                                       num_layers=num_layers,
                                       growth_rate=[cnn_channels // 2,
                                                    cnn_channels // 2,
                                                    cnn_channels // 2,
                                                    cnn_channels // 2,
                                                    cnn_channels // 2],
                                       drop_rate=0.1)
        self.swin_transfomer_encoder = SwinTransformerEncoder(embed_dim=swin_trans_channels,
                                                              drop_rate=0.1,
                                                              attn_drop_rate=0.1,
                                                              drop_path_rate=0.1)
        self.decoder = Decoder(depths=4,
                               num_classes=num_classes,
                               input_channels=cnn_channels,
                               vit_channels=swin_trans_channels,
                               cnn_channels=cnn_channels,
                               out_channels=cnn_channels)
        self.decoder_deep_sup = Decoder_Deep_Supervison(num_classes, cnn_channels)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        cnn_features = self.cnn_encoder(x)
        swin_transfomer_features = self.swin_transfomer_encoder(x)
        pred, res = self.decoder(swin_transfomer_features, cnn_features)
        deep_sup_out = self.decoder_deep_sup(res)
        return pred, deep_sup_out

class Decoder_Deep_Supervison(nn.Module):
    def __init__(self, num_classes, cnn_channels):
        super().__init__()
        self.feature_map_10 = nn.Sequential(
            SeparableConv(in_channels=cnn_channels * (2 ** 1), out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=cnn_channels),
            nn.ReLU(inplace=True)
        )
        self.feature_map_11 = SeparableConv(in_channels=cnn_channels, out_channels=num_classes, kernel_size=1)
        self.feature_map_20 = nn.Sequential(
            SeparableConv(in_channels=cnn_channels * (2 ** 2), out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=cnn_channels),
            nn.ReLU(inplace=True)
        )
        self.feature_map_21 = SeparableConv(in_channels=cnn_channels, out_channels=num_classes, kernel_size=1)
        self.feature_map_30 = nn.Sequential(
            SeparableConv(in_channels=cnn_channels * (2 ** 3), out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=cnn_channels),
            nn.ReLU(inplace=True)
        )
        self.feature_map_31 = SeparableConv(in_channels=cnn_channels, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        out3 = self.feature_map_30(x[0])
        out3 = F.interpolate(out3, size=(224, 224), mode='bilinear')
        out3 = self.feature_map_31(out3)

        out2 = self.feature_map_20(x[1])
        out2 = F.interpolate(out2, size=(224, 224), mode='bilinear')
        out2 = self.feature_map_21(out2)

        out1 = self.feature_map_10(x[2])
        out1 = F.interpolate(out1, size=(224, 224), mode='bilinear')
        out1 = self.feature_map_11(out1)

        return out1, out2, out3


if __name__ == "__main__":
    input_features = torch.randn(4, 3, 224, 224)
    network = CaT_Net_with_Decoder_DeepSup(num_classes=2,
                                           cnn_channels=32,
                                           swin_trans_channels=24,
                                           num_layers=[4, 4, 4, 4, 4])
    pred, res = network(input_features)
    print(pred.shape)