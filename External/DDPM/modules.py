import torch
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
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.last_attn = None  # <-- store last attention

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        x_ln = self.ln(x)

        attn_out, attn_weights = self.mha(
            x_ln, x_ln, x_ln, need_weights=True, average_attn_weights=False
        )

        # attn_weights: [B, n_heads, HW, HW]
        attn_weights = attn_weights.mean(dim=1)  # moyenne sur les têtes → [B, HW, HW]

        self.last_attn = attn_weights.detach()
        self.last_H = H
        self.last_W = W

        x = attn_out + x
        x = self.ff_self(x) + x
        return x.permute(0, 2, 1).view(B, C, H, W)


class SelfAttention4(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.last_attn = None  # <-- store last attention

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x_ln = self.ln(x)

        attn_out, attn_weights = self.mha(
            x_ln, x_ln, x_ln, need_weights=True, average_attn_weights=False
        )

        # attn_weights: [B, n_heads, HW, HW]
        attn_weights = attn_weights.mean(dim=1)  # moyenne sur les têtes → [B, HW, HW]

        self.last_attn = attn_weights.detach()
        self.last_H = H
        self.last_W = W

        x = attn_out + x
        x = self.ff_self(x) + x
        return x.permute(0, 2, 1).view(B, C, H, W)


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
    def __init__(self, in_ch, out_ch, time_dim, cond_dims):
        super().__init__()
        self.time_dim = time_dim
        self.cond_dims = cond_dims

        # remove fixed scale factor
        self.conv = DoubleConv(in_ch, out_ch)
        self.emb_layer = nn.Linear(time_dim, out_ch)

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
        # --- robust upsample to match skip size ---
        x = F.interpolate(x, size=x_skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x_skip, x], dim=1)
        x = self.conv(x)

        # --- Time embedding ---
        t_emb = UNet_conditional_256.get_time_embedding(self, t, self.time_dim)
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
    def __init__(
        self,
        c_in=1,
        c_out=1,
        cond_dims=None,
        image_size=256,
        time_dim=128,
        attention_from=32,
        attention_to=8,
        attention_on="both",  # "down", "up", "both"
    ):
        """
        Conditional UNet with configurable self-attention resolutions and flexible attention placement.

        Args:
            c_in (int): input channels
            c_out (int): output channels
            cond_dims (dict): conditioning description
            image_size (int): input image resolution (square)
            time_dim (int): timestep embedding dimension
            attention_from (int): max resolution for self-attention
            attention_to (int): min resolution for self-attention
            attention_on (str): where to apply attention ("down", "up", "both")
        """
        super().__init__()
        self.cond_dims = cond_dims
        self.time_dim = time_dim
        self.attention_on = attention_on

        # ---------------------------
        # Attention helper functions
        # ---------------------------
        def make_attention(channels, resolution):
            """Decide which self-attention layer to use for a feature map."""
            n_heads = 2 if channels <= 64 else 4
            min_res = n_heads  # minimal size to apply attention
            if resolution < min_res:
                return nn.Identity()
            if attention_to <= resolution <= attention_from:
                return (
                    SelfAttention2(channels)
                    if n_heads == 2
                    else SelfAttention4(channels)
                )
            return nn.Identity()

        def maybe_attention_down(channels, resolution):
            return (
                make_attention(channels, resolution)
                if attention_on in ["down", "both"]
                else nn.Identity()
            )

        def maybe_attention_up(channels, resolution):
            return (
                make_attention(channels, resolution)
                if attention_on in ["up", "both"]
                else nn.Identity()
            )

        # ---------------------------
        # Initial conv
        # ---------------------------
        self.inc = DoubleConv(c_in, 16)
        res = image_size

        # ---------------------------
        # Downsampling path
        # ---------------------------
        self.down1 = Down(16, 32, self.time_dim, cond_dims)
        res //= 2
        self.sa1 = maybe_attention_down(32, res)

        self.down2 = Down(32, 64, self.time_dim, cond_dims)
        res //= 2
        self.sa2 = maybe_attention_down(64, res)

        self.down3 = Down(64, 128, self.time_dim, cond_dims)
        res //= 2
        self.sa3 = maybe_attention_down(128, res)

        self.down4 = Down(128, 256, self.time_dim, cond_dims)
        res //= 2
        self.sa4 = maybe_attention_down(256, res)

        self.down5 = Down(256, 512, self.time_dim, cond_dims)
        res //= 2
        self.sa5 = maybe_attention_down(512, res)

        self.down6 = Down(512, 512, self.time_dim, cond_dims)
        res //= 2
        self.sa6 = maybe_attention_down(512, res)

        # ---------------------------
        # Bottleneck
        # ---------------------------
        self.bot1 = DoubleConv(512, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 512)

        # ---------------------------
        # Upsampling path
        # ---------------------------
        self.up6 = Up(1024, 256, self.time_dim, cond_dims)
        self.as6 = maybe_attention_up(256, res * 2)

        self.up5 = Up(512, 128, self.time_dim, cond_dims)
        self.as5 = maybe_attention_up(128, res * 4)

        self.up4 = Up(256, 64, self.time_dim, cond_dims)
        self.as4 = maybe_attention_up(64, res * 8)

        self.up3 = Up(128, 32, self.time_dim, cond_dims)
        self.as3 = maybe_attention_up(32, res * 16)

        self.up2 = Up(64, 16, self.time_dim, cond_dims)
        self.as2 = maybe_attention_up(16, res * 32)

        self.up1 = Up(32, 8, self.time_dim, cond_dims)
        self.as1 = maybe_attention_up(8, res * 64)

        # ---------------------------
        # Output
        # ---------------------------
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

    def forward(self, x, t, CT, WR, COMP, MFR, MAG, debug=False):
        B = x.size(0)
        t = t.view(B).float()

        x0 = self.inc(x)
        if debug:
            print("inc:", x0.min().item(), x0.max().item())
        x1 = self.sa1(self.down1(x0, t, CT, WR, COMP, MFR, MAG))
        if debug:
            print("down1:", x1.min().item(), x1.max().item())
        x2 = self.sa2(self.down2(x1, t, CT, WR, COMP, MFR, MAG))
        if debug:
            print("down2:", x2.min().item(), x2.max().item())
        x3 = self.sa3(self.down3(x2, t, CT, WR, COMP, MFR, MAG))
        if debug:
            print("down3:", x3.min().item(), x3.max().item())
        x4 = self.sa4(self.down4(x3, t, CT, WR, COMP, MFR, MAG))
        if debug:
            print("down4:", x4.min().item(), x4.max().item())
        x5 = self.sa5(self.down5(x4, t, CT, WR, COMP, MFR, MAG))
        if debug:
            print("down5:", x5.min().item(), x5.max().item())
        x6 = self.sa6(self.down6(x5, t, CT, WR, COMP, MFR, MAG))
        if debug:
            print("down6:", x6.min().item(), x6.max().item())

        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)
        if debug:
            print("bottleneck:", x6.min().item(), x6.max().item())

        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)

        x = self.as6(self.up6(x6, x5, t, CT, WR, COMP, MFR, MAG))
        x = self.as5(self.up5(x, x4, t, CT, WR, COMP, MFR, MAG))
        x = self.as4(self.up4(x, x3, t, CT, WR, COMP, MFR, MAG))
        x = self.as3(self.up3(x, x2, t, CT, WR, COMP, MFR, MAG))
        x = self.as2(self.up2(x, x1, t, CT, WR, COMP, MFR, MAG))
        x = self.as1(self.up1(x, x0, t, CT, WR, COMP, MFR, MAG))

        return self.outc(x)

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

    def get_last_attention_maps(self):
        """
        Collect only attention modules that are actually applied (not Identity)
        and have stored attention weights.
        """
        attn_modules = {}
        for name, module in self.named_modules():
            if isinstance(module, (SelfAttention2, SelfAttention4)) and hasattr(
                module, "last_attn"
            ):
                # Only include modules that are active given attention_on
                if module is not None:
                    attn_modules[name] = module
        return attn_modules


class UNet_conditional_256(nn.Module):
    def __init__(
        self,
        c_in=1,
        c_out=1,
        cond_dims=None,
        image_size=256,
        time_dim=128,
        attention_from=64,
        attention_to=16,
        attention_on="both",  # "down", "up", "both"
    ):
        """
        Conditional UNet for 256x256 patches with reduced width and selective attention.
        """
        super().__init__()
        self.cond_dims = cond_dims
        self.time_dim = time_dim
        self.attention_on = attention_on

        # ---------------------------
        # Attention helper functions
        # ---------------------------
        def make_attention(channels, resolution):
            n_heads = 2 if channels <= 64 else 4
            min_res = n_heads
            if resolution < min_res:
                return nn.Identity()
            if attention_to <= resolution <= attention_from:
                return (
                    SelfAttention2(channels)
                    if n_heads == 2
                    else SelfAttention4(channels)
                )
            return nn.Identity()

        def maybe_attention_down(channels, resolution):
            return (
                make_attention(channels, resolution)
                if attention_on in ["down", "both"]
                else nn.Identity()
            )

        def maybe_attention_up(channels, resolution):
            return (
                make_attention(channels, resolution)
                if attention_on in ["up", "both"]
                else nn.Identity()
            )

        # ---------------------------
        # Initial conv
        # ---------------------------
        self.inc = DoubleConv(c_in, 16)
        res = image_size

        # ---------------------------
        # Downsampling path
        # ---------------------------
        self.down1 = Down(16, 32, self.time_dim, cond_dims)
        res //= 2
        self.sa1 = maybe_attention_down(32, res)

        self.down2 = Down(32, 64, self.time_dim, cond_dims)
        res //= 2
        self.sa2 = maybe_attention_down(64, res)

        self.down3 = Down(64, 128, self.time_dim, cond_dims)
        res //= 2
        self.sa3 = maybe_attention_down(128, res)

        self.down4 = Down(128, 128, self.time_dim, cond_dims)
        res //= 2
        self.sa4 = maybe_attention_down(128, res)

        self.down5 = Down(128, 128, self.time_dim, cond_dims)
        res //= 2
        self.sa5 = maybe_attention_down(128, res)

        # ---------------------------
        # Bottleneck
        # ---------------------------
        self.bot1 = DoubleConv(128, 128)
        self.bot2 = DoubleConv(128, 128)

        # ---------------------------
        # Upsampling path
        # ---------------------------
        self.up5 = Up(256, 128, self.time_dim, cond_dims)
        self.as5 = maybe_attention_up(128, res * 2)

        self.up4 = Up(256, 128, self.time_dim, cond_dims)
        self.as4 = maybe_attention_up(128, res * 4)

        self.up3 = Up(256, 64, self.time_dim, cond_dims)
        self.as3 = maybe_attention_up(64, res * 8)

        self.up2 = Up(128, 32, self.time_dim, cond_dims)
        self.as2 = maybe_attention_up(32, res * 16)

        self.up1 = Up(48, 16, self.time_dim, cond_dims)  # x0 + up2_out = 16+32=48 ✅
        self.as1 = maybe_attention_up(16, res * 32)

        # ---------------------------
        # Output
        # ---------------------------
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)

    # ---------------------------
    # Time embedding helpers
    # ---------------------------
    def pos_encoding(self, t, channels):
        device = t.device
        t = t.view(-1)
        freqs = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t[:, None] * freqs)
        pos_enc_b = torch.cos(t[:, None] * freqs)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.unsqueeze(-1).unsqueeze(-1)

    def get_time_embedding(self, t, time_dim):
        device = t.device
        B = t.size(0)
        t = t.view(B, 1).float()
        half_dim = time_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    # ---------------------------
    # Forward
    # ---------------------------
    def forward(self, x, t, CT, WR, COMP, MFR, MAG, debug=False):
        B = x.size(0)
        t = t.view(B).float()

        # Initial conv
        x0 = self.inc(x)

        # Downsampling
        x1 = self.sa1(self.down1(x0, t, CT, WR, COMP, MFR, MAG))
        x2 = self.sa2(self.down2(x1, t, CT, WR, COMP, MFR, MAG))
        x3 = self.sa3(self.down3(x2, t, CT, WR, COMP, MFR, MAG))
        x4 = self.sa4(self.down4(x3, t, CT, WR, COMP, MFR, MAG))
        x5 = self.sa5(self.down5(x4, t, CT, WR, COMP, MFR, MAG))

        # Bottleneck
        xb = self.bot1(x5)
        xb = self.bot2(xb)

        # Upsampling
        x = self.as5(self.up5(xb, x5, t, CT, WR, COMP, MFR, MAG))
        x = self.as4(self.up4(x, x4, t, CT, WR, COMP, MFR, MAG))
        x = self.as3(self.up3(x, x3, t, CT, WR, COMP, MFR, MAG))
        x = self.as2(self.up2(x, x2, t, CT, WR, COMP, MFR, MAG))
        x = self.as1(self.up1(x, x0, t, CT, WR, COMP, MFR, MAG))

        return self.outc(x)

    # ---------------------------
    # Attention map helper
    # ---------------------------
    def get_last_attention_maps(self):
        attn_modules = {}
        for name, module in self.named_modules():
            if isinstance(module, (SelfAttention2, SelfAttention4)) and hasattr(
                module, "last_attn"
            ):
                attn_modules[name] = module
        return attn_modules
