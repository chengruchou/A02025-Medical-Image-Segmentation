import torch
import torch.nn.functional as F

from .metrics import dice_coef

def dice_loss(y_true, y_pred):
    """
    Calculate dice loss.
    """
    return 1. - dice_coef(y_true, y_pred)

def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel
    """
    coords = torch.arange(size).float() - size // 2
    coords = coords.unsqueeze(0).repeat(size, 1)
    gaussian = torch.exp(-(coords ** 2 + coords.t() ** 2) / (2 * sigma ** 2))
    return gaussian / gaussian.sum()

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = gaussian_kernel(window_size, 1.5).unsqueeze(0).unsqueeze(0)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = gaussian_kernel(self.window_size, 1.5).unsqueeze(0).unsqueeze(0)
            window = window.repeat(channel, 1, 1, 1)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)

def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index loss.
    Input shape should be Batch x #Classes x Height x Width (BxCxHxW).
    """
    ssim = SSIM()
    return ssim(y_true, y_pred)

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error loss.
    """
    return F.mse_loss(y_true, y_pred)


def criterion(y_true, y_pred):
    """
    Combined loss function.
    """
    return dice_loss(y_true, y_pred) + ssim_loss(y_true, y_pred) + mse_loss(y_true, y_pred)