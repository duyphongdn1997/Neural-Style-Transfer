from typing import Any

import torch
from torch import nn

from neural_style_transfer.model.feature_extractor import FeatureExtractor
from neural_style_transfer.utils import get_gram


class Trainer:
    def __init__(self, optimizer: Any, content_weight: int = 1, style_weight: int = 100):

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.optimizer = optimizer

    def train(self,
              model: FeatureExtractor,
              epochs: int,
              content_img: Any,
              style_img: Any,
              generated_img: Any,
              verbose: bool = True):
        for epoch in range(epochs):

            content_features = model(content_img)
            style_features = model(style_img)
            generated_features = model(generated_img)

            content_loss = torch.mean((content_features[-1] - generated_features[-1]) ** 2)

            style_loss = 0
            for gf, sf in zip(generated_features, style_features):
                _, c, h, w = gf.size()
                gram_gf = get_gram(gf)
                gram_sf = get_gram(sf)
                style_loss += torch.mean((gram_gf - gram_sf) ** 2) / (c * h * w)

            loss = self.content_weight * content_loss + self.style_weight * style_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0 and verbose:
                print('Epoch [{}]\tContent Loss: {:.4f}\tStyle Loss: {:.4f}'.format(epoch, content_loss.item(),
                                                                                    style_loss.item()))
        return model
