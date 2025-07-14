import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3DfMRI(nn.Module):
    def _initialize_weights(self, module):
        import torch.nn.init as init
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm3d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)

    def __init__(self, input_channels=1, conv_filt=32, kernel_size=5, activation="relu", pool_size=2):
        super().__init__()
        padding = kernel_size // 2
        act_fn  = nn.ReLU() if activation=="relu" else nn.LeakyReLU(0.2)

        # Encoder
        self.enc1 = self._block(input_channels,   conv_filt,    kernel_size, padding, act_fn)
        self.pool1 = nn.MaxPool3d(pool_size)
        self.enc2 = self._block(conv_filt,        2*conv_filt,  kernel_size, padding, act_fn)
        self.pool2 = nn.MaxPool3d(pool_size)
        self.enc3 = self._block(2*conv_filt,      4*conv_filt,  kernel_size, padding, act_fn)
        self.pool3 = nn.MaxPool3d(pool_size)
        self.enc4 = self._block(4*conv_filt,      8*conv_filt,  kernel_size, padding, act_fn)
        self.pool4 = nn.MaxPool3d(pool_size)

        # Bottleneck
        self.bottleneck = self._block(8*conv_filt, 16*conv_filt, kernel_size, padding, act_fn)

        # Decoder
        self.up4 = nn.ConvTranspose3d(16*conv_filt,  8*conv_filt, kernel_size=2, stride=2)
        self.dec4 = self._block(16*conv_filt,  8*conv_filt, kernel_size, padding, act_fn)

        self.up3 = nn.ConvTranspose3d(8*conv_filt,   4*conv_filt, kernel_size=2, stride=2)
        self.dec3 = self._block(8*conv_filt,   4*conv_filt, kernel_size, padding, act_fn)

        self.up2 = nn.ConvTranspose3d(4*conv_filt,   2*conv_filt, kernel_size=2, stride=2)
        self.dec2 = self._block(4*conv_filt,   2*conv_filt, kernel_size, padding, act_fn)

        self.up1 = nn.ConvTranspose3d(2*conv_filt,   conv_filt,   kernel_size=2, stride=2)
        self.dec1 = self._block(2*conv_filt,   conv_filt,   kernel_size, padding, act_fn)

        self.output_conv = nn.Conv3d(conv_filt, 1, kernel_size=1)
        self.apply(self._initialize_weights)

    def _block(self, in_ch, out_ch, k, p, act_fn):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, padding=p),
            nn.BatchNorm3d(out_ch),
            act_fn,
            nn.Conv3d(out_ch, out_ch, k, padding=p),
            nn.BatchNorm3d(out_ch),
            act_fn,
        )

    def forward(self, x):
        # compute padding so D,H,W become multiples of 16
        factor = 16
        _,_, D, H, W = x.shape
        pad_d = (factor - (D % factor)) % factor
        pad_h = (factor - (H % factor)) % factor
        pad_w = (factor - (W % factor)) % factor

        # correct order: (W_left, W_right, H_left, H_right, D_left, D_right)
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        orig = (D, H, W)

        # Encoder
        c1 = self.enc1(x); p1 = self.pool1(c1)
        c2 = self.enc2(p1); p2 = self.pool2(c2)
        c3 = self.enc3(p2); p3 = self.pool3(c3)
        c4 = self.enc4(p3); p4 = self.pool4(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        #Decoder (direct concat—no cropping)
        u4 = self.up4(bn);   d4 = self.dec4(torch.cat([c4, u4], dim=1))
        u3 = self.up3(d4);   d3 = self.dec3(torch.cat([c3, u3], dim=1))
        u2 = self.up2(d3);   d2 = self.dec2(torch.cat([c2, u2], dim=1))
        u1 = self.up1(d2);   d1 = self.dec1(torch.cat([c1, u1], dim=1))

        out = self.output_conv(d1)

        #  un‑pad back to original D,H,W
        D0, H0, W0 = orig
        return out[..., :D0, :H0, :W0]