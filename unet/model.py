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

    def __init__(self, input_channels=1, conv_filt=32, kernel_size=5, activation="relu", pool_size=2):
        super(UNet3DfMRI, self).__init__()
        padding = kernel_size // 2
        act_fn = nn.ReLU() if activation == "relu" else nn.LeakyReLU(0.2)

        # Encoder blocks
        self.enc1 = self._block(input_channels, conv_filt, kernel_size, padding, act_fn)
        self.pool1 = nn.MaxPool3d(pool_size)

        self.enc2 = self._block(conv_filt, 2*conv_filt, kernel_size, padding, act_fn)
        self.pool2 = nn.MaxPool3d(pool_size)

        self.enc3 = self._block(2*conv_filt, 4*conv_filt, kernel_size, padding, act_fn)
        self.pool3 = nn.MaxPool3d(pool_size)

        self.enc4 = self._block(4*conv_filt, 8*conv_filt, kernel_size, padding, act_fn)
        self.pool4 = nn.MaxPool3d(pool_size)

        # Bottleneck
        self.bottleneck = self._block(8*conv_filt, 16*conv_filt, kernel_size, padding, act_fn)

        # Decoder blocks
        self.up4 = nn.ConvTranspose3d(16*conv_filt, 8*conv_filt, kernel_size=2, stride=2)
        self.dec4 = self._block(16*conv_filt, 8*conv_filt, kernel_size, padding, act_fn)

        self.up3 = nn.ConvTranspose3d(8*conv_filt, 4*conv_filt, kernel_size=2, stride=2)
        self.dec3 = self._block(8*conv_filt, 4*conv_filt, kernel_size, padding, act_fn)

        self.up2 = nn.ConvTranspose3d(4*conv_filt, 2*conv_filt, kernel_size=2, stride=2)
        self.dec2 = self._block(4*conv_filt, 2*conv_filt, kernel_size, padding, act_fn)

        self.up1 = nn.ConvTranspose3d(2*conv_filt, conv_filt, kernel_size=2, stride=2)
        self.dec1 = self._block(2*conv_filt, conv_filt, kernel_size, padding, act_fn)

        self.output_conv = nn.Conv3d(conv_filt, 1, kernel_size=1)
        self.apply(self._initialize_weights)

    def _block(self, in_channels, out_channels, kernel_size, padding, act_fn):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            act_fn,
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm3d(out_channels),
            act_fn
        )

    def forward(self, x):
        # Encoding
        conv1 = self.enc1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.enc2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.enc3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.enc4(pool3)
        pool4 = self.pool4(conv4)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Decoding
        up4 = self.up4(bottleneck)
        conv4_cropped = self.crop_to_match(conv4, up4)
        dec4 = self.dec4(torch.cat([conv4_cropped, up4], dim=1))

        up3 = self.up3(dec4)
        conv3_cropped = self.crop_to_match(conv3, up3)
        dec3 = self.dec3(torch.cat([conv3_cropped, up3], dim=1))

        up2 = self.up2(dec3)
        conv2_cropped = self.crop_to_match(conv2, up2)
        dec2 = self.dec2(torch.cat([conv2_cropped, up2], dim=1))

        up1 = self.up1(dec2)
        conv1_cropped = self.crop_to_match(conv1, up1)
        dec1 = self.dec1(torch.cat([conv1_cropped, up1], dim=1))

        return self.output_conv(dec1)
