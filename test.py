import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision.models
from timm.models.swin_transformer_v2 import SwinTransformerV2Stage
from timm.layers.helpers import to_2tuple

from WeatherLearn.weatherlearn.models import Fuxi

from era5_dataset import ERA5Dataset

ds = ERA5Dataset('C://Users//lukas//PycharmProjects//DL-Weather-Forecasting//data//era5_6hourly//era5_6hourly.zarr', 32)

dataset_iter = iter(ds)


class FuXi(nn.Module):
    def __init__(self, img_size, input_channels, output_channels):
        super().__init__()

        self.img_size = img_size
        self.patch_size = (1, 2, 2)
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.window_size = to_2tuple(7)

        self.conv3d = nn.Conv3d(input_channels, output_channels, self.patch_size, self.patch_size)
        self.layernorm = nn.LayerNorm(output_channels)

        _downblock = []
        _downblock.append(nn.Conv2d(output_channels, output_channels, 3, 1, 1))
        _downblock.append(nn.GroupNorm(output_channels, output_channels))
        _downblock.append(nn.SiLU())
        self.downblock = nn.Sequential(*_downblock)

        self.upscaling = nn.Upsample(scale_factor=1.5, mode="bilinear", align_corners=True)

        self.swin = SwinTransformerV2Stage(output_channels, output_channels, (90, 180), 2, 1, 5)

        _upblock = []
        _upblock.append(nn.Conv2d(output_channels, output_channels, 3, 1, 1))
        _upblock.append(nn.GroupNorm(output_channels, output_channels))
        _upblock.append(nn.SiLU())
        self.upblock = nn.Sequential(*_upblock)

        self.fc = nn.Linear(output_channels, output_channels*2*2)

        self.downscaling = nn.Upsample(scale_factor=4/3, mode="bilinear", align_corners=True)


    def forward(self, x):
        batch, values, h, l, b = x.shape

        x = self.conv3d(x)
        _size = tuple(x[0][0].shape)
        print("CubeEmbedding -> Conv3d abgeschlossen, Shape des Tensors: ", x.shape)
        x = x.reshape(batch, self.output_channels, -1).transpose(1, 2)
        x = self.layernorm(x)
        x = x.transpose(1, 2).reshape(batch, self.output_channels, *_size)
        print("CubeEmbedding -> LayerNorm abgeschlossen, Shape des Tensors: ", x.shape)
        x = x.mean(dim=2)
        x = self.upscaling(x)
        print("UpScaling -> UpScaling abgeschlossen, Shape des Tensors: ", x.shape)
        x = self.downblock(x)
        print("DownBlock -> DownBlock abgeschlossen, Shape des Tensors: ", x.shape)

        x = x.permute(0, 2, 3, 1)
        x = self.swin(x)
        x = x.permute(0, 3, 1, 2)
        print("SwinTransformer -> SwinTransformer abgeschlossen, Shape des Tensors: ", x.shape)

        x = self.upblock(x)
        print("UpBlock -> UpBlock abgeschlossen, Shape des Tensors: ", x.shape)

        x = x.permute(0, 2, 3, 1)
        x = self.fc(x)
        x = x.reshape(batch, 180, 360, self.output_channels)
        x = x.permute(0, 3, 1, 2)
        print("FCLayer -> FCLayer abgeschlossen, Shape des Tensors: ", x.shape)

        #x = self.downscaling(x)
        #print("DownScaling -> DownScaling abgeschlossen, Shape des Tensors: ", x.shape)

        x = F.interpolate(x, size=(121, 240), mode="bilinear")
        print("Interpolate -> Interpolate abgeschlossen, Shape des Tensors: ", x.shape)

        return x



#_fuxi = FuXi(img_size=(1, 121, 240), input_channels=5, output_channels=5)
#print("Start von FuXi, Shape des Tensors: ", t.shape)
#t = _fuxi(t)
#print("Ende von FuXi, Shape des Tensors: ", t.shape)

lossfunction = nn.L1Loss()
model = FuXi((1, 121, 240), 5, 5)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):
    print("Epoche: ", epoch)
    inputs, labels = next(dataset_iter)
    inputs = inputs[:, :, 0, :, :].unsqueeze(2)
    labels = labels[:, :, 0, :, :]
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = lossfunction(outputs, labels)
    loss.backward()
    optimizer.step()
