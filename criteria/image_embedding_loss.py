import torch
import clip
import torchvision.transforms as transforms

class ImageEmbddingLoss(torch.nn.Module):

    def __init__(self):
        super(ImageEmbddingLoss, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.cosloss = torch.nn.CosineEmbeddingLoss()

    def forward(self, masked_generated, masked_img_tensor):
        masked_generated = self.face_pool(masked_generated)
        masked_generated_renormed = self.transform(masked_generated * 0.5 + 0.5)
        masked_generated_feature = self.model.encode_image(masked_generated_renormed)

        masked_img_tensor = self.face_pool(masked_img_tensor)
        masked_img_tensor_renormed = self.transform(masked_img_tensor * 0.5 + 0.5)
        masked_img_tensor_feature = self.model.encode_image(masked_img_tensor_renormed)

        cos_target = torch.ones((masked_img_tensor.shape[0], 1)).float().cuda()
        similarity = self.cosloss(masked_generated_feature, masked_img_tensor_feature, cos_target).unsqueeze(0).unsqueeze(0)
        return similarity
