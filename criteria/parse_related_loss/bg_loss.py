import torch
from torch import nn
from criteria.parse_related_loss.unet import unet

class BackgroundLoss(nn.Module):
    def __init__(self, opts):
        super(BackgroundLoss, self).__init__()
        print('Loading UNet for Background Loss')
        self.parsenet = unet()
        self.parsenet.load_state_dict(torch.load(opts.parsenet_weights))
        self.parsenet.eval()
        self.bg_mask_l2_loss = torch.nn.MSELoss()
        self.shrink = torch.nn.AdaptiveAvgPool2d((512, 512))
        self.magnify = torch.nn.AdaptiveAvgPool2d((1024, 1024))
         

    def gen_bg_mask(self, input_image):
        labels_predict = self.parsenet(self.shrink(input_image)).detach()
        mask_512 = (torch.unsqueeze(torch.max(labels_predict, 1)[1], 1)!=13).float()
        mask_1024 = self.magnify(mask_512)
        return mask_1024

    def forward(self, x, x_hat):
        x_bg_mask = self.gen_bg_mask(x)
        x_hat_bg_mask = self.gen_bg_mask(x_hat)
        bg_mask = ((x_bg_mask+x_hat_bg_mask)==2).float()
        loss = self.bg_mask_l2_loss(x * bg_mask, x_hat * bg_mask)
        return loss

