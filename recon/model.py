from torch import nn
import segmentation_models_pytorch as smp
import torch


class ChangeNet(nn.Module):

    def __init__(self, model_name: str = "resnet16"):
        super().__init__()

        model = smp.Unet(model_name, encoder_depth=3, decoder_channels=[64, 64, 16])  # TODO move params to config

        self.encoder = model.encoder
        self.decoder = model.decoder
        self.head = model.segmentation_head

    def forward(self, image_1, image_2):
        enc1 = self.encoder(image_1)
        enc2 = self.encoder(image_2)

        encoder_output = []

        for i in range(len(enc1)):
            encoder_output.append(torch.add(enc1[i], enc2[i]))  # todo check sizes

        decoder_output = self.decoder(*encoder_output)

        segmentation_result = self.head(decoder_output)

        return segmentation_result
