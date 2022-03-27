import sys
import tempfile
from argparse import Namespace

import dlib
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from cog import BasePredictor, Path, Input

sys.path.insert(0, "encoder4editing")
from models.psp import pSp
from utils.alignment import align_face

sys.path.insert(0, "criteria/parse_related_loss")
import average_lab_color_loss
from mapper.datasets.latents_dataset_inference import LatentsDatasetInference
from mapper.hairclip_mapper import HairCLIPMapper


with open("mapper/hairstyle_list.txt") as infile:
    HAIRSTYLE_LIST = sorted([line.rstrip() for line in infile])


class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda:0"

        # use e4e to get latent code for an input image
        e4e_model_path = "pretrained_models/e4e_ffhq_encode.pt"
        e4e_ckpt = torch.load(e4e_model_path, map_location="cpu")
        e4e_opts = e4e_ckpt["opts"]
        e4e_opts["checkpoint_path"] = e4e_model_path
        e4e_opts = Namespace(**e4e_opts)

        self.e4e_net = pSp(e4e_opts)
        self.e4e_net.eval()
        self.e4e_net.cuda()
        print("e4e model successfully loaded!")

        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # HairClip model
        checkpoint_path = "pretrained_models/hairclip.pt"
        self.ckpt = torch.load(checkpoint_path, map_location="cpu")

    def predict(
        self,
        image: Path = Input(
            description="Input image. Image will be aligned and resized. Output will be the "
            "concatenation of the inverted input and the image with edited hair."
        ),
        editing_type: str = Input(
            choices=["hairstyle", "color", "both"],
            default="hairstyle",
            description="Edit hairstyle or color or both.",
        ),
        hairstyle_description: str = Input(
            choices=HAIRSTYLE_LIST,
            default=None,
            description="Hairstyle text prompt. "
            "Valid if input_type is text or text_image.",
        ),
        color_description: str = Input(
            default=None,
            description="Color text prompt, eg: purple, red, orange. "
            "Valid if editing_type is color or both.",
        ),
    ) -> Path:

        editing_type_ = str(editing_type).split(".")[-1]
        hairstyle_description_ = str(hairstyle_description).split(".")[-1]

        if editing_type_ == "both":
            assert (
                hairstyle_description_ is not None and color_description is not None
            ), ("Please provide description " "for both hairstyle and color.")
        elif editing_type_ == "hairstyle":
            assert (
                hairstyle_description_ is not None
            ), "Please provide description for hairstyle."
        else:
            assert (
                color_description is not None
            ), "Please provide description for color."

        opts = self.ckpt["opts"]
        opts = Namespace(**opts)
        opts.editing_type = editing_type_
        opts.input_type = "text"
        opts.color_description = color_description

        if hairstyle_description is not None:
            with open("hairstyle_description.txt", "w") as file:
                file.write(hairstyle_description_)

            opts.hairstyle_description = "hairstyle_description.txt"

        opts.checkpoint_path = "pretrained_models/hairclip.pt"
        opts.parsenet_weights = "pretrained_models/parsenet.pth"
        opts.stylegan_weights = "pretrained_models/stylegan2-ffhq-config-f.pt"
        opts.ir_se50_weights = "pretrained_models/model_ir_se50.pth"
        net = HairCLIPMapper(opts)
        net.eval()
        net.cuda()

        # first align, resize image and get latent code
        input_image = run_alignment(str(image))
        resize_dims = (256, 256)
        input_image.resize(resize_dims)
        transformed_image = self.img_transforms(input_image)

        with torch.no_grad():

            images, latents = run_on_batch_e4e(
                transformed_image.unsqueeze(0), self.e4e_net
            )
            print("Latent code calculated!")

        dataset = LatentsDatasetInference(latents=latents.cpu(), opts=opts)
        dataloader = DataLoader(dataset)

        average_color_loss = (
            average_lab_color_loss.AvgLabLoss(opts).to(self.device).eval()
        )

        out_path = Path(tempfile.mkdtemp()) / "output.png"

        for input_batch in tqdm(dataloader):

            with torch.no_grad():

                (
                    w,
                    hairstyle_text_inputs_list,
                    color_text_inputs_list,
                    selected_description_tuple_list,
                    hairstyle_tensor_list,
                    color_tensor_list,
                ) = input_batch
                hairstyle_text_inputs = hairstyle_text_inputs_list[0]
                color_text_inputs = color_text_inputs_list[0]
                selected_description = selected_description_tuple_list[0][0]
                hairstyle_tensor = hairstyle_tensor_list[0]
                color_tensor = color_tensor_list[0]
                w = w.cuda().float()
                hairstyle_text_inputs = hairstyle_text_inputs.cuda()
                color_text_inputs = color_text_inputs.cuda()
                hairstyle_tensor = hairstyle_tensor.cuda()
                color_tensor = color_tensor.cuda()
                if hairstyle_tensor.shape[1] != 1:
                    hairstyle_tensor_hairmasked = (
                        hairstyle_tensor
                        * average_color_loss.gen_hair_mask(hairstyle_tensor)
                    )
                else:
                    hairstyle_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
                if color_tensor.shape[1] != 1:
                    color_tensor_hairmasked = (
                        color_tensor * average_color_loss.gen_hair_mask(color_tensor)
                    )
                else:
                    color_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
                result_batch = run_on_batch(
                    w,
                    hairstyle_text_inputs,
                    color_text_inputs,
                    hairstyle_tensor_hairmasked,
                    color_tensor_hairmasked,
                    net,
                )

                if (hairstyle_tensor.shape[1] != 1) and (color_tensor.shape[1] != 1):
                    img_tensor = torch.cat([hairstyle_tensor, color_tensor], dim=3)
                elif hairstyle_tensor.shape[1] != 1:
                    img_tensor = hairstyle_tensor
                elif color_tensor.shape[1] != 1:
                    img_tensor = color_tensor
                else:
                    img_tensor = None

                if img_tensor is not None:
                    if img_tensor.shape[3] == 1024:
                        couple_output = torch.cat(
                            [
                                result_batch[2][0].unsqueeze(0),
                                result_batch[0][0].unsqueeze(0),
                                img_tensor,
                            ]
                        )
                    elif img_tensor.shape[3] == 2048:
                        couple_output = torch.cat(
                            [
                                result_batch[2][0].unsqueeze(0),
                                result_batch[0][0].unsqueeze(0),
                                img_tensor[:, :, :, 0:1024],
                                img_tensor[:, :, :, 1024::],
                            ]
                        )
                        couple_output = torch.cat(
                            [
                                result_batch[2][0].unsqueeze(0),
                                result_batch[0][0].unsqueeze(0),
                                img_tensor[:, :, :, 0:1024],
                                img_tensor[:, :, :, 1024::],
                            ]
                        )
                else:
                    couple_output = torch.cat(
                        [
                            result_batch[2][0].unsqueeze(0),
                            result_batch[0][0].unsqueeze(0),
                        ]
                    )
                torchvision.utils.save_image(
                    couple_output, str(out_path), normalize=True, range=(-1, 1)
                )

        return out_path


def run_alignment(image_path):
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image


def run_on_batch_e4e(inputs, net):
    images, latents = net(
        inputs.to("cuda").float(), randomize_noise=False, return_latents=True
    )
    return images, latents


def run_on_batch(
    inputs,
    hairstyle_text_inputs,
    color_text_inputs,
    hairstyle_tensor_hairmasked,
    color_tensor_hairmasked,
    net,
):
    w = inputs
    with torch.no_grad():
        w_hat = w + 0.1 * net.mapper(
            w,
            hairstyle_text_inputs,
            color_text_inputs,
            hairstyle_tensor_hairmasked,
            color_tensor_hairmasked,
        )
        x_hat, w_hat = net.decoder(
            [w_hat],
            input_is_latent=True,
            return_latents=True,
            randomize_noise=False,
            truncation=1,
        )
        x, _ = net.decoder(
            [w], input_is_latent=True, randomize_noise=False, truncation=1
        )
        result_batch = (x_hat, w_hat, x)
    return result_batch
