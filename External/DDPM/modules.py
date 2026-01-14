import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention2(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention2, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )


class SelfAttention4(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention4, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(
            -1, self.channels, self.size, self.size
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dims):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
        self.time_dim = time_dim
        self.emb_layer = nn.Linear(time_dim, out_ch)

        # Conditional embeddings
        self.cond_embeddings = nn.ModuleDict()
        if cond_dims is not None:
            for k, n_levels in cond_dims["num_levels"].items():
                self.cond_embeddings[k] = nn.Embedding(
                    n_levels, cond_dims["embedding_dim"]
                )
            self.cond_proj = nn.Linear(
                len(cond_dims["num_levels"]) * cond_dims["embedding_dim"], out_ch
            )

    def forward(self, x, t, CT=None, WR=None, COMP=None, MFR=None, MAG=None):
        x = self.maxpool_conv(x)

        # --- Time embedding ---
        t_emb = UNet_conditional.get_time_embedding(
            self, t, self.time_dim
        )  # [B, time_dim]
        t_emb = self.emb_layer(t_emb)  # [B, out_ch]

        # --- Conditional embedding ---
        cond_emb = 0
        if hasattr(self, "cond_embeddings") and CT is not None:
            embeds = []
            for k, v in zip(
                ["CT", "WR", "COMP", "MFR", "MAG"], [CT, WR, COMP, MFR, MAG]
            ):
                embeds.append(self.cond_embeddings[k](v))
            cond_emb = torch.cat(embeds, dim=1)
            cond_emb = self.cond_proj(cond_emb)  # [B, out_ch]

        # Combine and broadcast to spatial dims
        emb = t_emb + cond_emb
        emb = emb[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        x = x + emb
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, cond_dims, bilinear=True):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dims = cond_dims

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

        # Time embedding
        self.emb_layer = nn.Linear(time_dim, out_ch)

        # Conditional embeddings
        if cond_dims is not None:
            self.cond_embeddings = nn.ModuleDict()
            for k, n_levels in cond_dims["num_levels"].items():
                self.cond_embeddings[k] = nn.Embedding(
                    n_levels, cond_dims["embedding_dim"]
                )
            self.cond_proj = nn.Linear(
                len(cond_dims["num_levels"]) * cond_dims["embedding_dim"], out_ch
            )

    def forward(self, x, x_skip, t, CT=None, WR=None, COMP=None, MFR=None, MAG=None):
        x = self.up(x)
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)

        # --- Time embedding ---
        t_emb = UNet_conditional.get_time_embedding(self, t, self.time_dim)
        t_emb = self.emb_layer(t_emb)

        # --- Conditional embedding ---
        cond_emb = 0
        if hasattr(self, "cond_embeddings") and CT is not None:
            embeds = []
            for k, v in zip(
                ["CT", "WR", "COMP", "MFR", "MAG"], [CT, WR, COMP, MFR, MAG]
            ):
                embeds.append(self.cond_embeddings[k](v))
            cond_emb = torch.cat(embeds, dim=1)
            cond_emb = self.cond_proj(cond_emb)

        emb = t_emb + cond_emb
        emb = emb[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1])
        x = x + emb
        return x


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16, 32)
        self.sa1 = SelfAttention4(32, 128)
        self.down2 = Down(32, 64)
        self.sa2 = SelfAttention4(64, 64)
        self.down3 = Down(64, 128)
        self.sa3 = SelfAttention4(128, 32)
        self.down4 = Down(128, 256)
        self.sa4 = SelfAttention4(256, 16)
        self.down5 = Down(256, 256)
        self.sa5 = SelfAttention4(256, 8)
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        self.up5 = Up(512, 128)
        self.as5 = SelfAttention4(128, 16)
        self.up4 = Up(256, 64)
        self.as4 = SelfAttention4(64, 32)
        self.up3 = Up(128, 32)
        self.as3 = SelfAttention4(32, 64)
        self.up2 = Up(64, 16)
        self.as2 = SelfAttention4(16, 128)
        self.up1 = Up(32, 8)
        self.as1 = SelfAttention4(8, 256)
        self.outc = nn.Conv2d(8, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float().to(device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x0 = self.inc(x)
        x1 = self.down1(x0, t)
        # x1 = self.sa1(x1)
        x2 = self.down2(x1, t)
        x2 = self.sa2(x2)
        x3 = self.down3(x2, t)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t)
        x4 = self.sa4(x4)
        x5 = self.down5(x4, t)
        x5 = self.sa5(x5)
        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up5(x5, x4, t)
        # del x5, x4
        x = self.as5(x)
        x = self.up4(x, x3, t)
        # del x3
        x = self.as4(x)
        x = self.up3(x, x2, t)
        # del x2
        x = self.as3(x)
        x = self.up2(x, x1, t)
        # del x1
        # x = self.as2(x)
        x = self.up1(x, x0, t)
        # del x0
        # x = self.as1(x)
        x = self.outc(x)
        return x


class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=512, cond_dims=None):
        """
        Conditional UNet with Down/Up blocks supporting arbitrary numbers of levels per condition.

        Args:
            c_in (int): input channels (image)
            c_out (int): output channels
            time_dim (int): dimension of time embedding
            cond_dims (dict): {'num_levels': {...}, 'embedding_dim': int}
        """
        super().__init__()
        self.time_dim = time_dim
        self.cond_dims = cond_dims

        # Initial conv
        self.inc = DoubleConv(c_in, 16)

        # Down path
        self.down1 = Down(16, 32, 256, cond_dims)
        self.sa1 = SelfAttention4(32, 256)
        self.down2 = Down(32, 64, 128, cond_dims)
        self.sa2 = SelfAttention4(64, 128)
        self.down3 = Down(64, 128, 64, cond_dims)
        self.sa3 = SelfAttention2(128, 64)
        self.down4 = Down(128, 256, 32, cond_dims)
        self.sa4 = SelfAttention4(256, 32)
        self.down5 = Down(256, 512, 16, cond_dims)
        self.sa5 = SelfAttention4(512, 16)
        self.down6 = Down(512, 512, 8, cond_dims)
        self.sa6 = SelfAttention4(512, 8)

        # Bottleneck
        self.bot1 = DoubleConv(512, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 512)

        # Upsampling
        self.up6 = Up(1024, 256, 16, cond_dims)
        self.as6 = SelfAttention4(256, 16)
        self.up5 = Up(512, 128, 32, cond_dims)
        self.as5 = SelfAttention4(128, 32)
        self.up4 = Up(256, 64, 64, cond_dims)
        self.as4 = SelfAttention4(64, 64)
        self.up3 = Up(128, 32, 128, cond_dims)
        self.as3 = SelfAttention2(32, 128)
        self.up2 = Up(64, 16, 256, cond_dims)
        self.as2 = SelfAttention4(16, 256)
        self.up1 = Up(32, 8, 512, cond_dims)
        self.as1 = SelfAttention4(8, 512)

        # Output conv
        self.outc = nn.Conv2d(8, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        """
        Sinusoidal positional encoding for diffusion timesteps.
        Returns [B, channels, 1, 1] for broadcasting in Down/Up blocks.
        """
        device = t.device
        t = t.view(-1)  # ensure shape (B,)
        freqs = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t[:, None] * freqs)
        pos_enc_b = torch.cos(t[:, None] * freqs)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.unsqueeze(-1).unsqueeze(-1)  # [B, channels, 1, 1]

    def forward(self, x, t, CT, WR, COMP, MFR, MAG):
        """
        Forward pass for conditional UNet.

        Args:
            x (Tensor): input image [B, C, H, W]
            t (Tensor): timestep [B]
            CT, WR, COMP, MFR, MAG (Tensor): conditioning indices [B]
        """
        B = x.size(0)

        # Ensure t is 1D
        t = t.view(B).float()

        # ---------------------------
        # Downsampling path
        # ---------------------------
        x0 = self.inc(x)  # initial conv

        x1 = self.down1(x0, t, CT, WR, COMP, MFR, MAG)
        x1 = self.sa1(x1)
        x2 = self.down2(x1, t, CT, WR, COMP, MFR, MAG)
        x2 = self.sa2(x2)
        x3 = self.down3(x2, t, CT, WR, COMP, MFR, MAG)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t, CT, WR, COMP, MFR, MAG)
        x4 = self.sa4(x4)
        x5 = self.down5(x4, t, CT, WR, COMP, MFR, MAG)
        x5 = self.sa5(x5)
        x6 = self.down6(x5, t, CT, WR, COMP, MFR, MAG)
        x6 = self.sa6(x6)

        # ---------------------------
        # Bottleneck
        # ---------------------------
        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)

        # ---------------------------
        # Upsampling path
        # ---------------------------
        x = self.up6(x6, x5, t, CT, WR, COMP, MFR, MAG)
        x = self.as6(x)
        x = self.up5(x, x4, t, CT, WR, COMP, MFR, MAG)
        x = self.as5(x)
        x = self.up4(x, x3, t, CT, WR, COMP, MFR, MAG)
        x = self.as4(x)
        x = self.up3(x, x2, t, CT, WR, COMP, MFR, MAG)
        x = self.as3(x)
        x = self.up2(x, x1, t, CT, WR, COMP, MFR, MAG)
        x = self.up1(x, x0, t, CT, WR, COMP, MFR, MAG)

        # ---------------------------
        # Output
        # ---------------------------
        out = self.outc(x)
        return out

    def get_time_embedding(self, t, time_dim):
        """
        Sinusoidal positional encoding for diffusion timesteps.
        Returns [B, time_dim]
        """
        device = t.device
        B = t.size(0)
        t = t.view(B, 1).float()  # [B,1]
        half_dim = time_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, time_dim]
        if time_dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return emb  # [B, time_dim]


class UNet_conditional_small(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=512, cond_dims=None):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dims = cond_dims

        # Initial conv
        self.inc = DoubleConv(c_in, 16)

        # Down path
        self.down1 = Down(16, 32, time_dim, cond_dims)  # x1: 32
        self.down2 = Down(32, 64, time_dim, cond_dims)  # x2: 64
        self.down3 = Down(64, 128, time_dim, cond_dims)  # x3: 128

        # Bottleneck
        self.bot1 = DoubleConv(128, 128)
        self.bot2 = DoubleConv(128, 128)

        # Up path — **in_ch = skip + current**
        self.up3 = Up(128 + 64, 64, time_dim, cond_dims)  # x3 + x2
        self.up2 = Up(64 + 32, 32, time_dim, cond_dims)  # result + x1
        self.up1 = Up(32 + 16, 16, time_dim, cond_dims)  # result + x0

        # Output conv
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)

    def forward(self, x, t, CT, WR, COMP, MFR, MAG):
        B = x.size(0)
        t = t.view(B).float()

        # Down
        x0 = self.inc(x)  # 16
        x1 = self.down1(x0, t, CT, WR, COMP, MFR, MAG)  # 32
        x2 = self.down2(x1, t, CT, WR, COMP, MFR, MAG)  # 64
        x3 = self.down3(x2, t, CT, WR, COMP, MFR, MAG)  # 128

        # Bottleneck
        x3 = self.bot1(x3)
        x3 = self.bot2(x3)

        # Up
        x = self.up3(x3, x2, t, CT, WR, COMP, MFR, MAG)  # 64
        x = self.up2(x, x1, t, CT, WR, COMP, MFR, MAG)  # 32
        x = self.up1(x, x0, t, CT, WR, COMP, MFR, MAG)  # 16

        out = self.outc(x)
        return out

    def get_time_embedding(self, t, time_dim):
        device = t.device
        B = t.size(0)
        t = t.view(B, 1).float()
        half_dim = time_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb
        )
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb
