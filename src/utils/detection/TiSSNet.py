import numpy as np
import torch
from torch import nn

class TiSSNet(nn.Module):
    def __init__(self):
        super().__init__()

        modules = []

        modules.append(nn.Conv2d(1, 16, kernel_size=(8, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(16, 16, kernel_size=(8, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(16, 16, kernel_size=(8, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=(2, 1)))

        modules.append(nn.Conv2d(16, 32, kernel_size=(5, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(32, 32, kernel_size=(5, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(32, 32, kernel_size=(5, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=(4, 1)))

        modules.append(nn.Conv2d(32, 64, kernel_size=(3, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(64, 64, kernel_size=(3, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(64, 64, kernel_size=(3, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=(4, 1)))

        modules.append(nn.Conv2d(64, 128, kernel_size=(2, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(128, 128, kernel_size=(2, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))
        modules.append(nn.Conv2d(128, 128, kernel_size=(2, 8), padding='same'))
        modules.append(nn.LeakyReLU(negative_slope=0.3, inplace=True))

        modules.append(nn.MaxPool2d(kernel_size=(4, 1)))

        modules.append(nn.Conv2d(128, 1, kernel_size=1))
        modules.append(nn.Sigmoid())

        modules.append(nn.Flatten(-3, -1))

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


def process_batch(batch, device, model):
    try:
        batch = np.array(batch)
    except:
        print("the array is not rectangular")
    batch = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        res = model(batch).cpu().numpy()
    return res