import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import torch


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, num_patch, tokens_mlp_dim, channels_mlp_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, tokens_mlp_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            FeedForward(hidden_dim, channels_mlp_dim, dropout)
        )

    def forward(self, x):

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MlpMixer(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim=256,
                 patch_size=16,
                 image_size=256,
                 num_classes=1000,
                 num_blocks=8,
                 tokens_mlp_dim=256,
                 channels_mlp_dim=2048,
                 **kwargs):
        super(MlpMixer, self).__init__()
        self.num_patch = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.mixer_blocks = nn.ModuleList([])

        for _ in range(num_blocks):
            self.mixer_blocks.append(
                MixerBlock(hidden_dim, self.num_patch, tokens_mlp_dim, channels_mlp_dim)
            )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

    def to_patch_embedding(self, x):
        return self.patch_embedding(x)

    def forward(self, x):
        x = self.to_patch_embedding(x)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = x.mean(dim=1)

        return self.mlp_head(x)


if __name__ == "__main__":
    mlp = MlpMixer(in_channels=3)
    inputs = torch.randn(1, 3, 256, 256)
    outputs = mlp(inputs)
    pass