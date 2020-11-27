#!python
# -*- coding: utf-8 -*-
import os
import yaml
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from feature_utils import Audio2Mel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model(mel2wav_path, device=get_default_device()):
    """
    Args:
        mel2wav_path (str or Path): path to the root folder of dumped text2mel
        device (str or torch.device): device to load the model
    """
    root = Path(mel2wav_path)
    with open(root / "args.yml", "r") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
    netG.load_state_dict(torch.load(root / "best_netG.pt", map_location=device))
    return netG


class MelVocoder:
    def __init__(
            self,
            path,
            device=get_default_device(),
            github=False,
            model_name="multi_speaker",
    ):
        self.fft = Audio2Mel().to(device)
        if github:
            netG = Generator(80, 32, 3).to(device)
            root = Path(os.path.dirname(__file__)).parent
            netG.load_state_dict(
                torch.load(root / f"models/{model_name}.pt", map_location=device)
            )
            self.mel2wav = netG
        else:
            self.mel2wav = load_model(path, device)
        self.device = device

    def __call__(self, audio):
        """
        Performs audio to mel conversion (See Audio2Mel in mel2wav/modules.py)
        Args:
            audio (torch.tensor): PyTorch tensor containing audio (batch_size, timesteps)
        Returns:
            torch.tensor: log-mel-spectrogram computed on input audio (batch_size, 80, timesteps)
        """
        return self.fft(audio.unsqueeze(1).to(self.device))

    def inverse(self, mel):
        """
        Performs mel2audio conversion
        Args:
            mel (torch.tensor): PyTorch tensor containing log-mel spectrograms (batch_size, 80, timesteps)
        Returns:
            torch.tensor:  Inverted raw audio (batch_size, timesteps)

        """
        with torch.no_grad():
            return self.mel2wav(mel.to(self.device)).squeeze(1)
