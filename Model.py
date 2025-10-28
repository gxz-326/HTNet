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

class DiagonalMicroAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.):
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Dropout(dropout)
        )
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.asymmetry_gate = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def compute_diagonal_attention(self, q, k, v, h_spatial, w_spatial):
        b, heads, n, d = q.shape
        
        attn_normal = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn_scores = self.attend(attn_normal)
        attn_scores = self.dropout(attn_scores)
        
        q_2d = rearrange(q, 'b h (x y) d -> b h x y d', x=h_spatial, y=w_spatial)
        k_2d = rearrange(k, 'b h (x y) d -> b h x y d', x=h_spatial, y=w_spatial)
        
        q_diag_main = torch.diagonal(q_2d, dim1=2, dim2=3)
        k_diag_main = torch.diagonal(k_2d, dim1=2, dim2=3)
        
        q_flipped = torch.flip(q_2d, [3])
        k_flipped = torch.flip(k_2d, [3])
        q_diag_anti = torch.diagonal(q_flipped, dim1=2, dim2=3)
        k_diag_anti = torch.diagonal(k_flipped, dim1=2, dim2=3)
        
        diag_attn_main = einsum('b h n d, b h m d -> b h n m', q_diag_main, k_diag_main) * self.scale
        diag_attn_anti = einsum('b h n d, b h m d -> b h n m', q_diag_anti, k_diag_anti) * self.scale
        
        diag_weight = 0.3
        enhanced_attn = attn_scores + diag_weight * (
            F.pad(torch.diagonal(diag_attn_main.mean(dim=1), dim1=-2, dim2=-1), 
                  (0, n - min(h_spatial, w_spatial))).unsqueeze(1).unsqueeze(-1)
        )
        
        out = einsum('b h i j, b h j d -> b h i d', enhanced_attn, v)
        return out
    
    def detect_left_right_asymmetry(self, x):
        b, c, h, w = x.shape
        mid = w // 2
        
        left_side = x[:, :, :, :mid]
        right_side = x[:, :, :, mid:]
        
        if right_side.shape[-1] != left_side.shape[-1]:
            right_side = F.pad(right_side, (0, left_side.shape[-1] - right_side.shape[-1]))
        
        right_flipped = torch.flip(right_side, [3])
        
        asymmetry = torch.abs(left_side - right_flipped)
        asymmetry_pooled = F.adaptive_avg_pool2d(asymmetry, (h, w))
        
        asymmetry_weight = self.asymmetry_gate(asymmetry_pooled)
        
        return asymmetry_weight
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=self.heads), qkv)
        
        out = self.compute_diagonal_attention(q, k, v, h, w)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        
        asymmetry_weight = self.detect_left_right_asymmetry(x)
        out = out * (1.0 + asymmetry_weight)
        
        return self.to_out(out)

class ROIAttentionModule(nn.Module):
    def __init__(self, dim, num_roi_regions=5):
        super().__init__()
        self.num_roi_regions = num_roi_regions
        
        self.roi_detector = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, num_roi_regions, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.BatchNorm2d(dim // 4),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )
        
        self.background_suppressor = nn.Sequential(
            nn.Conv2d(dim + num_roi_regions, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.roi_refine = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        roi_maps = self.roi_detector(x)
        
        roi_aggregated = torch.sum(roi_maps, dim=1, keepdim=True)
        roi_mask = (roi_aggregated > 0.3).float()
        
        spatial_att = self.spatial_attention(x)
        spatial_att = spatial_att * roi_mask
        
        channel_att = self.channel_attention(x)
        
        attended = x * spatial_att * channel_att
        
        combined = torch.cat([attended, roi_maps], dim=1)
        suppressed = self.background_suppressor(combined)
        
        refined = self.roi_refine(suppressed)
        
        output = x + refined
        
        return output, roi_maps

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

def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding = 1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride = 2, padding = 1)
    )

class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dropout = dropout)),
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

class EnhancedTransformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0., 
                 use_diagonal_attn=True, use_roi=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))
        self.use_diagonal_attn = use_diagonal_attn
        self.use_roi = use_roi

        for i in range(depth):
            layer_modules = nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ])
            
            if use_diagonal_attn and i % 2 == 0:
                layer_modules.append(PreNorm(dim, DiagonalMicroAttention(dim, heads=heads//2, dropout=dropout)))
            
            if use_roi and i == depth - 1:
                layer_modules.append(ROIAttentionModule(dim, num_roi_regions=5))
            
            self.layers.append(layer_modules)
    
    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h=h, w=w)
        x = x + pos_emb

        roi_maps_list = []
        
        for layer_modules in self.layers:
            attn = layer_modules[0]
            ff = layer_modules[1]
            
            x = attn(x) + x
            x = ff(x) + x
            
            if len(layer_modules) > 2:
                if isinstance(layer_modules[2], PreNorm):
                    diag_attn = layer_modules[2]
                    x = diag_attn(x) + x
                
                if len(layer_modules) > 3 and isinstance(layer_modules[3], ROIAttentionModule):
                    roi_module = layer_modules[3]
                    x, roi_maps = roi_module(x)
                    roi_maps_list.append(roi_maps)
        
        return x, roi_maps_list if roi_maps_list else None

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
        dropout = 0.
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

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))


        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1 = block_size, b2 = block_size)
            x = aggregate(x)
        return self.mlp_head(x)

class HTNetEnhanced(nn.Module):
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
        mlp_mult=4,
        channels=3,
        dim_head=64,
        dropout=0.,
        use_diagonal_attn=True,
        use_roi=True
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2
        fmap_size = image_size // patch_size
        blocks = 2 ** (num_hierarchies - 1)

        seq_len = (fmap_size // blocks) ** 2
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        self.use_diagonal_attn = use_diagonal_attn
        self.use_roi = use_roi
        
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                EnhancedTransformer(dim_in, seq_len, depth, heads, mlp_mult, dropout,
                                   use_diagonal_attn=use_diagonal_attn,
                                   use_roi=use_roi and is_last),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        self.mlp_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, img, return_roi_maps=False):
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        all_roi_maps = []
        
        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x, roi_maps = transformer(x)
            if roi_maps is not None:
                all_roi_maps.extend(roi_maps)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x = aggregate(x)
        
        output = self.mlp_head(x)
        
        if return_roi_maps:
            return output, all_roi_maps
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
