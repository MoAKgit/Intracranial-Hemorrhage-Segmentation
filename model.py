# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:45:24 2023

@author: Mohammad
"""





import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"






class Double2DConv(nn.Module):
    """two serial 2D CNN"""

    def __init__(
        self,
        in_channels,
        out_channels,
        batchNorm = True,
        kernel_size = [3, 3],
        stride_size = 1,
        padding = 1,
        activation = nn.ReLU(inplace=True),
    ):
        """
        this class create two serial 2D CNN

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            batchNorm (boolean, optuional): have batchnormalization
            layer or not
            kernel_size (list[int,int], optional): Defaults to [3, 3].
            stride_size (int, optional): Defaults to 1.
            padding (int, optional): Defaults to 1.
            activation (torch.nn.modules.activation, optional): specific
            activation function. Defaults to nn.ReLU(inplace=True).
        """
        super(Double2DConv, self).__init__()
        # TODO: remove sequential form of this class
        
        
        if batchNorm:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride_size,
                    padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                activation,
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride_size,
                    padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                activation,
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride_size,
                    padding,
                    bias=False,
                ),
                activation,
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride_size,
                    padding,
                    bias=False,
                ),
                activation,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """copmute the output of 2 cnn layers"""
        return self.conv(x)

class UNET2D(nn.Module):
    """basic Unet network implementation based on the below paper
    https://doi.org/10.48550/arXiv.1505.04597"""

    # TODO: reconstruct with decoder and encoder classes see xnet class
    def __init__(
        self,
        in_channels,
        out_channels,
        # features: list[int] = [64, 128, 256, 512]
        # TODO: add kernel padding stride batchnorm
    ) -> None:
        """this class contatin structure of 2-D U_NET network

        Args:
            in_channels (int, optional): Defaults to 3.
            out_channels (int, optional):  Defaults to 3.
            features (list[int], optional): Defaults to [64, 128, 256, 512].
        """
        super(UNET2D, self).__init__()
        
        features = [64, 128, 256, 512]
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        # Down part of UNET
        for feature in features:
            self.downs.append(Double2DConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(
                Double2DConv(feature * 2, feature)
            )  # two cnn on top

        self.bottleneck = Double2DConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=3,
                                    padding=1)
        
    def forward(self, x):
        """generate a segmentation image

        Args:
            x (tensor): the image that would be segmented

        Returns:
            segmented image
        """
        skip_connections = []
        
        
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)
        # TODO: make change in code to mak this class modular
        # check number of classes to segment and toggle softmax
        # check kind of loss function and toggle sigmoid
        # get input to print output shape or not
        # x = self.softmax(x)
        # print(x.shape)
        # x = torch.clamp(x, min= 0, max= 1)
        x = self.sigmoid(x)
        # x = self.relu(x)

        return x