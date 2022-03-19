from torch.utils.data import Dataset
import numpy as np
import clip
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
from training import train_utils
import os

class LatentsDatasetInference(Dataset):
	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts

		if self.opts.editing_type in ['hairstyle', 'both'] and self.opts.input_type.split('_')[0] == 'text':
			with open(self.opts.hairstyle_description, "r") as fd:
				self.hairstyle_description_list = fd.read().splitlines()
			self.hairstyle_list = [single_hairstyle_description[:-9] for single_hairstyle_description in self.hairstyle_description_list]
		if self.opts.editing_type in ['color', 'both'] and self.opts.input_type.split('_')[-1] == 'text':
			self.color_list = [single_color_description.strip()+' ' for single_color_description in self.opts.color_description.split(',')]
		if self.opts.editing_type in ['hairstyle', 'both'] and self.opts.input_type.split('_')[0] == 'image':
			self.out_domain_hairstyle_img_path_list = sorted(train_utils.make_dataset(self.opts.hairstyle_ref_img_test_path))
		if self.opts.editing_type in ['color', 'both'] and self.opts.input_type.split('_')[-1] == 'image':
			self.out_domain_color_img_path_list = sorted(train_utils.make_dataset(self.opts.color_ref_img_test_path))

		self.image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


	def manipulate_hairstyle(self, index):
		if self.opts.input_type.split('_')[0] == 'text':
			color_text_embedding_list = [torch.Tensor([0]) for i in range(len(self.hairstyle_list))]
			color_tensor_list = [torch.Tensor([0]) for i in range(len(self.hairstyle_list))]
			hairstyle_tensor_list = [torch.Tensor([0]) for i in range(len(self.hairstyle_list))]
			selected_hairstyle_description_list = [single_hairstyle_description+'hairstyle' for single_hairstyle_description in self.hairstyle_list]
			hairstyle_text_embedding_list = [torch.cat([clip.tokenize(selected_hairstyle_description)])[0] for selected_hairstyle_description in selected_hairstyle_description_list]
		elif self.opts.input_type.split('_')[0] == 'image':
			color_text_embedding_list = [torch.Tensor([0]) for i in range(self.opts.num_of_ref_img)]
			color_tensor_list = [torch.Tensor([0]) for i in range(self.opts.num_of_ref_img)]
			hairstyle_text_embedding_list = [torch.Tensor([0]) for i in range(self.opts.num_of_ref_img)]
			selected_hairstyle_description_list = ['hairstyle_out_domain_ref' for i in range(self.opts.num_of_ref_img)]
			hairstyle_tensor_list = [self.image_transform(Image.open(random.choice(self.out_domain_hairstyle_img_path_list))) for i in range(self.opts.num_of_ref_img)]
		return self.latents[index], hairstyle_text_embedding_list, color_text_embedding_list, selected_hairstyle_description_list, hairstyle_tensor_list, color_tensor_list


	def manipulater_color(self, index):
		if self.opts.input_type.split('_')[-1] == 'text':
			hairstyle_text_embedding_list = [torch.Tensor([0]) for i in range(len(self.color_list))]
			hairstyle_tensor_list = [torch.Tensor([0]) for i in range(len(self.color_list))]
			color_tensor_list = [torch.Tensor([0]) for i in range(len(self.color_list))]
			selected_color_description_list = [single_color_description+'hair' for single_color_description in self.color_list]
			color_text_embedding_list = [torch.cat([clip.tokenize(selected_color_description)])[0] for selected_color_description in selected_color_description_list]
		elif self.opts.input_type.split('_')[-1] == 'image':
			hairstyle_text_embedding_list = [torch.Tensor([0]) for i in range(self.opts.num_of_ref_img)]
			hairstyle_tensor_list = [torch.Tensor([0]) for i in range(self.opts.num_of_ref_img)]
			color_text_embedding_list = [torch.Tensor([0]) for i in range(self.opts.num_of_ref_img)]
			selected_color_description_list = ['color_out_domain_ref' for i in range(self.opts.num_of_ref_img)]
			color_tensor_list = [self.image_transform(Image.open(random.choice(self.out_domain_color_img_path_list))) for i in range(self.opts.num_of_ref_img)]
		return self.latents[index], hairstyle_text_embedding_list, color_text_embedding_list, selected_color_description_list, hairstyle_tensor_list, color_tensor_list		


	def manipulater_hairstyle_and_color(self, index):
		returned_latent, hairstyle_text_embedding_list, _, selected_hairstyle_description_list, hairstyle_tensor_list, _ = self.manipulate_hairstyle(index)
		_, _, color_text_embedding_list, selected_color_description_list, _, color_tensor_list = self.manipulater_color(index)
		hairstyle_text_embedding_final_list = [hairstyle_text_embedding for hairstyle_text_embedding in hairstyle_text_embedding_list for i in color_text_embedding_list]
		color_text_embedding_final_list = [color_text_embedding for i in hairstyle_text_embedding_list for color_text_embedding in color_text_embedding_list]
		selected_description_list = [f'{selected_hairstyle_description}-{selected_color_description}' for selected_hairstyle_description in selected_hairstyle_description_list for selected_color_description in selected_color_description_list]
		hairstyle_tensor_final_list = [hairstyle_tensor for hairstyle_tensor in hairstyle_tensor_list for i in color_tensor_list]
		color_tensor_final_list = [color_tensor for i in hairstyle_tensor_list for color_tensor in color_tensor_list]
		return returned_latent, hairstyle_text_embedding_final_list, color_text_embedding_final_list, selected_description_list, hairstyle_tensor_final_list, color_tensor_final_list


	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		if self.opts.editing_type == 'hairstyle':
			return self.manipulate_hairstyle(index)
		elif self.opts.editing_type == 'color':
			return self.manipulater_color(index)
		elif self.opts.editing_type == 'both':
			return self.manipulater_hairstyle_and_color(index)



