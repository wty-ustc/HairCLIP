import os
from argparse import Namespace
import sys
sys.path.append(".")
sys.path.append("..")
import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from criteria.parse_related_loss import average_lab_color_loss
from tqdm import tqdm
from mapper.datasets.latents_dataset_inference import LatentsDatasetInference
from mapper.options.test_options import TestOptions
from mapper.hairclip_mapper import HairCLIPMapper


def run(test_opts):
	device = 'cuda:0'
	out_path_results = os.path.join(test_opts.exp_dir, test_opts.editing_type, test_opts.input_type)
	os.makedirs(out_path_results, exist_ok=True)
	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	opts = Namespace(**opts)
	net = HairCLIPMapper(opts)
	net.eval()
	net.cuda()

	test_latents = torch.load(opts.latents_test_path)
	dataset = LatentsDatasetInference(latents=test_latents.cpu(),
										 opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)
	average_color_loss = average_lab_color_loss.AvgLabLoss(opts).to(device).eval()
	assert (opts.start_index >= 0) and (opts.end_index <= len(dataset))
	global_i = 0
	for input_batch in tqdm(dataloader):
		if global_i not in range(opts.start_index, opts.end_index):
			if global_i >=opts.end_index:
				break
			global_i += 1
			continue
		with torch.no_grad():
			w, hairstyle_text_inputs_list, color_text_inputs_list, selected_description_tuple_list, hairstyle_tensor_list, color_tensor_list = input_batch
			for i in range(len(selected_description_tuple_list)):
				hairstyle_text_inputs = hairstyle_text_inputs_list[i]
				color_text_inputs = color_text_inputs_list[i]
				selected_description = selected_description_tuple_list[i][0]
				hairstyle_tensor = hairstyle_tensor_list[i]
				color_tensor = color_tensor_list[i]
				w = w.cuda().float()
				hairstyle_text_inputs = hairstyle_text_inputs.cuda()
				color_text_inputs = color_text_inputs.cuda()
				hairstyle_tensor = hairstyle_tensor.cuda()
				color_tensor = color_tensor.cuda()
				if hairstyle_tensor.shape[1] != 1:
					hairstyle_tensor_hairmasked = hairstyle_tensor * average_color_loss.gen_hair_mask(hairstyle_tensor)
				else:
					hairstyle_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
				if color_tensor.shape[1] != 1:
					color_tensor_hairmasked = color_tensor * average_color_loss.gen_hair_mask(color_tensor)
				else:
					color_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
				result_batch = run_on_batch(w, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor_hairmasked, color_tensor_hairmasked, net)
				
				if (hairstyle_tensor.shape[1] != 1) and (color_tensor.shape[1] != 1):
					img_tensor = torch.cat([hairstyle_tensor, color_tensor], dim = 3)
				elif hairstyle_tensor.shape[1] != 1:
					img_tensor = hairstyle_tensor
				elif color_tensor.shape[1] != 1:
					img_tensor = color_tensor
				else:
					img_tensor = None

				im_path = str(global_i).zfill(5)
				if img_tensor is not None:
					if img_tensor.shape[3] == 1024:
						couple_output = torch.cat([result_batch[2][0].unsqueeze(0), result_batch[0][0].unsqueeze(0), img_tensor])
					elif img_tensor.shape[3] == 2048:
						couple_output = torch.cat([result_batch[2][0].unsqueeze(0), result_batch[0][0].unsqueeze(0), img_tensor[:,:,:,0:1024], img_tensor[:,:,:,1024::]])
				else:
					couple_output = torch.cat([result_batch[2][0].unsqueeze(0), result_batch[0][0].unsqueeze(0)])
				torchvision.utils.save_image(couple_output, os.path.join(out_path_results, f"{im_path}-{str(i).zfill(4)}-{selected_description}.jpg"), normalize=True, range=(-1, 1))
			global_i += 1


def run_on_batch(inputs, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor_hairmasked, color_tensor_hairmasked, net):
	w = inputs
	with torch.no_grad():
		w_hat = w + 0.1 * net.mapper(w, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor_hairmasked, color_tensor_hairmasked)
		x_hat, w_hat = net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
		result_batch = (x_hat, w_hat)
		x, _ = net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
		result_batch = (x_hat, w_hat, x)
	return result_batch


if __name__ == '__main__':
	test_opts = TestOptions().parse()
	if test_opts.test_batch_size != 1:
		raise Exception('This version only supports test batch size to be 1.')	
	run(test_opts)
