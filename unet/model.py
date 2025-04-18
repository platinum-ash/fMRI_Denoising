import torch
import torch.nn as nn


class UNet3DfMRI(nn.Module):

    @classmethod
    def crop_to_match(cls, src, target):
        """Center-crop src to match the shape of target"""
        src_shape = src.shape[2:]
        target_shape = target.shape[2:]
        crop = [(s - t) // 2 for s, t in zip(src_shape, target_shape)]
        return src[
            :,
            :,
            crop[0]:crop[0]+target_shape[0],
            crop[1]:crop[1]+target_shape[1],
            crop[2]:crop[2]+target_shape[2],
        ]
    
    
    def _initialize_weights(self, module):
        import torch.nn.init as init
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm3d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)


    def __init__(self, input_channels=1, conv_filt=32, kernel_size=3, activation="relu", pool_size=2):
        super(UNet3DfMRI, self).__init__()
        padding = kernel_size // 2
        act_fn = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(input_channels, conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(conv_filt),
            act_fn,
            nn.Conv3d(conv_filt, conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(conv_filt),
            act_fn
        )
        self.pool1 = nn.MaxPool3d(pool_size)

        self.enc2 = nn.Sequential(
            nn.Conv3d(conv_filt, 2*conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(2*conv_filt),
            act_fn,
            nn.Conv3d(2*conv_filt, 2*conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(2*conv_filt),
            act_fn
        )
        self.pool2 = nn.MaxPool3d(pool_size)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(2*conv_filt, 4*conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(4*conv_filt),
            act_fn,
            nn.Conv3d(4*conv_filt, 2*conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(2*conv_filt),
            act_fn
        )

        # Decoder
        self.up1 = nn.ConvTranspose3d(2*conv_filt, 2*conv_filt, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(4*conv_filt, 2*conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(2*conv_filt),
            act_fn,
            nn.Conv3d(2*conv_filt, conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(conv_filt),
            act_fn
        )

        self.up2 = nn.ConvTranspose3d(conv_filt, conv_filt, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(2*conv_filt, conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(conv_filt),
            act_fn,
            nn.Conv3d(conv_filt, conv_filt, kernel_size, padding=padding),
            nn.BatchNorm3d(conv_filt),
            act_fn
        )

        self.output_conv = nn.Conv3d(conv_filt, 1, kernel_size=1)
        self.apply(self._initialize_weights)


    def forward(self, x):
        # Encoder
        conv1 = self.enc1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.enc2(pool1)
        pool2 = self.pool2(conv2)

        # Bottleneck
        bottleneck = self.bottleneck(pool2)

        # Decoder
        up1 = self.up1(bottleneck)
        conv2_cropped = self.crop_to_match(conv2, up1)
        merge1 = torch.cat([conv2_cropped, up1], dim=1)
        dec1 = self.dec1(merge1)

        up2 = self.up2(dec1)
        conv1_cropped = self.crop_to_match(conv1, up2)
        merge2 = torch.cat([conv1_cropped, up2], dim=1)
        dec2 = self.dec2(merge2)

        return self.output_conv(dec2)
