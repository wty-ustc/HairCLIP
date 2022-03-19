from torch.utils.data import Dataset
import numpy as np
import clip
import torch
import random
from PIL import Image
import torchvision.transforms as transforms
from training import train_utils
import os
class LatentsDataset(Dataset):

	def __init__(self, latents, opts, status='train'):
		self.latents = latents
		self.opts = opts
		self.status = status
		assert (self.opts.hairstyle_manipulation_prob+self.opts.color_manipulation_prob+self.opts.both_manipulation_prob) <= 1
		with open(self.opts.hairstyle_description, "r") as fd:
			self.hairstyle_description_list = fd.read().splitlines()

		self.hairstyle_list = [single_hairstyle_description[:-9] for single_hairstyle_description in self.hairstyle_description_list]
		self.color_list = [single_color_description.strip()+' ' for single_color_description in self.opts.color_description.split(',')]
		self.image_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		if self.status == 'train':
			self.out_domain_hairstyle_img_path_list = sorted(train_utils.make_dataset(self.opts.hairstyle_ref_img_train_path))
			self.out_domain_color_img_path_list = sorted(train_utils.make_dataset(self.opts.color_ref_img_train_path))
		else:
			self.out_domain_hairstyle_img_path_list = sorted(train_utils.make_dataset(self.opts.hairstyle_ref_img_test_path))
			self.out_domain_color_img_path_list = sorted(train_utils.make_dataset(self.opts.color_ref_img_test_path))


	def manipulate_hairstyle(self, index):
		color_text_embedding = torch.Tensor([0])
		color_tensor = torch.Tensor([0])
		if random.random() < self.opts.hairstyle_text_manipulation_prob:
			selected_hairstyle_description = np.random.choice(self.hairstyle_list)+'hairstyle'
			selected_description = selected_hairstyle_description
			hairstyle_text_embedding = torch.cat([clip.tokenize(selected_hairstyle_description)])[0]
			hairstyle_tensor = torch.Tensor([0])
		else:
			hairstyle_text_embedding = torch.Tensor([0])
			img_pil = Image.open(random.choice(self.out_domain_hairstyle_img_path_list))
			hairstyle_tensor = self.image_transform(img_pil)
			selected_description = 'hairstyle_out_domain_ref'
		return self.latents[index], hairstyle_text_embedding, color_text_embedding, selected_description, hairstyle_tensor, color_tensor

	def manipulater_color(self, index):
		hairstyle_text_embedding = torch.Tensor([0])
		hairstyle_tensor = torch.Tensor([0])
		selected_color_description = np.random.choice(self.color_list)+'hair'
		if random.random() < self.opts.color_text_manipulation_prob:
			selected_description = selected_color_description
			color_text_embedding = torch.cat([clip.tokenize(selected_color_description)])[0]
			color_tensor = torch.Tensor([0])
		else:
			color_text_embedding = torch.Tensor([0])
			if random.random() < (self.opts.color_in_domain_ref_manipulation_prob/(1-self.opts.color_text_manipulation_prob)):
				selected_description = 'color_in_domain_ref'
				img_pil = Image.open(self.opts.color_ref_img_in_domain_path+selected_color_description+'/'+str(random.randint(0, (self.opts.num_for_each_augmented_color-1))).zfill(5)+'.jpg')
				color_tensor = self.image_transform(img_pil)
			else:
				selected_description = 'color_out_domain_ref'
				img_pil = Image.open(random.choice(self.out_domain_color_img_path_list))
				color_tensor = self.image_transform(img_pil)
		return self.latents[index], hairstyle_text_embedding, color_text_embedding, selected_description, hairstyle_tensor, color_tensor

	def manipulater_hairstyle_and_color(self, index):
		returned_latent, hairstyle_text_embedding, _, selected_hairstyle_description, hairstyle_tensor, _ = self.manipulate_hairstyle(index)
		_, _, color_text_embedding, selected_color_description, _, color_tensor = self.manipulater_color(index)
		selected_description = f'{selected_hairstyle_description}-{selected_color_description}'
		return returned_latent, hairstyle_text_embedding, color_text_embedding, selected_description, hairstyle_tensor, color_tensor

	def no_editing(self, index):
		return self.latents[index], torch.Tensor([0]), torch.Tensor([0]), 'no_editing', torch.Tensor([0]), torch.Tensor([0])

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		function_list = ['self.manipulate_hairstyle(index)', 'self.manipulater_color(index)', 'self.manipulater_hairstyle_and_color(index)', 'self.no_editing(index)']
		prob_array = np.array([self.opts.hairstyle_manipulation_prob, self.opts.color_manipulation_prob, self.opts.both_manipulation_prob, (1-self.opts.hairstyle_manipulation_prob-self.opts.color_manipulation_prob-self.opts.both_manipulation_prob)])
		return eval(np.random.choice(function_list, replace=False, p=prob_array.ravel()))



