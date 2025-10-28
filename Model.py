from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

class DiagonalMicroAttention(nn.Module):
    def __init__(self, dim, heads = 4, dropout = 0., asymmetry_weight = 0.5):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.asymmetry_weight = asymmetry_weight

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        self.asymmetry_score = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, 1, 1),
            nn.Sigmoid()
        )

    def compute_asymmetry_map(self, x):
        b, c, h, w = x.shape
        
        left_half = x[:, :, :, :w//2]
        right_half = x[:, :, :, w//2:]
        right_half_flipped = torch.flip(right_half, [3])
        
        if left_half.shape[3] != right_half_flipped.shape[3]:
            min_width = min(left_half.shape[3], right_half_flipped.shape[3])
            left_half = left_half[:, :, :, :min_width]
            right_half_flipped = right_half_flipped[:, :, :, :min_width]
        
        concat_features = torch.cat([left_half, right_half_flipped], dim=1)
        asymmetry_map = self.asymmetry_score(concat_features)
        
        asymmetry_map_full = F.interpolate(asymmetry_map, size=(h, w), mode='bilinear', align_corners=False)
        
        return asymmetry_map_full

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads
        
        asymmetry_map = self.compute_asymmetry_map(x)
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), [q, k, v])
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        diagonal_mask = torch.zeros(h * w, h * w, device=x.device)
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            nidx = ni * w + nj
                            diagonal_mask[idx, nidx] = 1.0
        
        diagonal_mask = diagonal_mask.unsqueeze(0).unsqueeze(0)
        dots = dots * diagonal_mask + (1 - diagonal_mask) * (-1e9)
        
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        
        out = out * (1 + self.asymmetry_weight * asymmetry_map)
        
        return self.to_out(out)

class FacialROIModule(nn.Module):
    def __init__(self, dim, num_roi_regions = 5):
        super().__init__()
        self.num_roi_regions = num_roi_regions
        
        self.roi_detector = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, num_roi_regions, 1),
            nn.Sigmoid()
        )
        
        self.background_suppressor = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        self.roi_refine = nn.Sequential(
            nn.Conv2d(dim + num_roi_regions + 1, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.register_buffer('facial_prior', self._create_facial_prior())
    
    def _create_facial_prior(self):
        prior = torch.ones(1, 1, 32, 32)
        h, w = 32, 32
        
        center_h, center_w = h // 2, w // 2
        
        for i in range(h):
            for j in range(w):
                dist = ((i - center_h) ** 2 + (j - center_w) ** 2) ** 0.5
                max_dist = ((h/2) ** 2 + (w/2) ** 2) ** 0.5
                prior[0, 0, i, j] = max(0, 1 - (dist / max_dist) * 0.5)
        
        return prior
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        roi_maps = self.roi_detector(x)
        
        bg_suppression = self.background_suppressor(x)
        
        facial_prior_resized = F.interpolate(self.facial_prior, size=(h, w), mode='bilinear', align_corners=False)
        facial_prior_resized = facial_prior_resized.expand(b, -1, -1, -1)
        
        combined_mask = bg_suppression * facial_prior_resized
        
        roi_weighted = roi_maps * combined_mask
        
        enhanced_features = torch.cat([x, roi_weighted, combined_mask], dim=1)
        refined_features = self.roi_refine(enhanced_features)
        
        output = x + refined_features * combined_mask
        
        return output, roi_weighted, combined_mask

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0., use_micro_attention = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))
        self.use_micro_attention = use_micro_attention

        for i in range(depth):
            if use_micro_attention and i % 2 == 1:
                attn_layer = PreNorm(dim, DiagonalMicroAttention(dim, heads = heads, dropout = dropout))
            else:
                attn_layer = PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))
            
            self.layers.append(nn.ModuleList([
                attn_layer,
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout = dropout))
            ]))
    
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h = h, w = w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class HTNet(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        heads,
        num_hierarchies,
        block_repeats,
        mlp_mult = 4,
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        use_micro_attention = False,
        use_roi_module = False,
        num_roi_regions = 5
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2 #
        fmap_size = image_size // patch_size #
        blocks = 2 ** (num_hierarchies - 1)#

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1 = patch_size, p2 = patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        self.use_roi_module = use_roi_module
        if use_roi_module:
            self.roi_module = FacialROIModule(layer_dims[0], num_roi_regions=num_roi_regions)

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout, use_micro_attention=use_micro_attention),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))


        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img, return_attention_maps=False):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        roi_maps = None
        roi_mask = None
        
        if self.use_roi_module:
            x, roi_maps, roi_mask = self.roi_module(x)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)
        
        output = self.mlp_head(x)
        
        if return_attention_maps:
            return output, {'roi_maps': roi_maps, 'roi_mask': roi_mask}
        return output

# This function is to confuse three models
class Fusionmodel(nn.Module):
  def __init__(self):
    #  extend from original
    super(Fusionmodel,self).__init__()
    self.fc1 = nn.Linear(15, 3)
    self.bn1 = nn.BatchNorm1d(3)
    self.d1 = nn.Dropout(p=0.5)
    self.fc_2 = nn.Linear(6, 3)
    self.relu = nn.ReLU()
    # forward layers is to use these layers above
  def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
    fuse_four_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
    fuse_out = self.fc1(fuse_four_features)
    fuse_out = self.relu(fuse_out)
    fuse_out = self.d1(fuse_out) # drop out
    fuse_whole_four_parts = torch.cat(
        (whole_feature,fuse_out), 0)
    fuse_whole_four_parts = self.relu(fuse_whole_four_parts)
    fuse_whole_four_parts = self.d1(fuse_whole_four_parts)
    out = self.fc_2(fuse_whole_four_parts)
    return out
