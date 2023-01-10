from torch import nn
import segmentation_models_pytorch as smp
import torch

from configs import ModelParameters


class ChangeNet(nn.Module):

    def __init__(self, model_params: ModelParameters):
        super().__init__()
        self._model_name = model_params.model_name
        self._encoder_depth = model_params.encoder_depth
        self._decoder_channels = model_params.decoder_channels
        model = smp.Unet(self._model_name, encoder_depth=self._encoder_depth,
                         decoder_channels=self._decoder_channels)

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

        return torch.sigmoid(segmentation_result)
