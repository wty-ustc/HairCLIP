import os
import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from criteria.parse_related_loss import bg_loss, average_lab_color_loss
import criteria.clip_loss as clip_loss
import criteria.image_embedding_loss as image_embedding_loss
from criteria import id_loss
from mapper.datasets.latents_dataset import LatentsDataset
from mapper.hairclip_mapper import HairCLIPMapper
from mapper.training.ranger import Ranger
from mapper.training import train_utils
class Coach:
	def __init__(self, opts):
		self.opts = opts
		self.global_step = 0
		self.device = 'cuda:0'
		self.opts.device = self.device

		# Initialize network
		self.net = HairCLIPMapper(self.opts).to(self.device)

		# Initialize loss
		self.id_loss = id_loss.IDLoss(self.opts).to(self.device).eval()
		self.clip_loss = clip_loss.CLIPLoss(opts)
		self.latent_l2_loss = nn.MSELoss().to(self.device).eval()
		self.background_loss = bg_loss.BackgroundLoss(self.opts).to(self.device).eval()
		self.image_embedding_loss = image_embedding_loss.ImageEmbddingLoss()
		self.average_color_loss = average_lab_color_loss.AvgLabLoss(self.opts).to(self.device).eval()
		self.maintain_color_for_hairstyle_loss = average_lab_color_loss.AvgLabLoss(self.opts).to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)



		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.log_dir = log_dir
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				w, hairstyle_text_inputs, color_text_inputs, selected_description_tuple, hairstyle_tensor, color_tensor = batch
				selected_description = ''
				for item in selected_description_tuple:
					selected_description+=item

				w = w.to(self.device)
				hairstyle_text_inputs = hairstyle_text_inputs.to(self.device)
				color_text_inputs = color_text_inputs.to(self.device)
				hairstyle_tensor = hairstyle_tensor.to(self.device)
				color_tensor = color_tensor.to(self.device)
				with torch.no_grad():
					x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
				if hairstyle_tensor.shape[1] != 1:
					hairstyle_tensor_hairmasked = hairstyle_tensor * self.average_color_loss.gen_hair_mask(hairstyle_tensor)
				else:
					hairstyle_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
				if color_tensor.shape[1] != 1:
					color_tensor_hairmasked = color_tensor * self.average_color_loss.gen_hair_mask(color_tensor)
				else:
					color_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
				w_hat = w + 0.1 * self.net.mapper(w, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor_hairmasked, color_tensor_hairmasked)
				x_hat, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
				loss, loss_dict = self.calc_loss(w, x, w_hat, x_hat, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor, color_tensor, selected_description)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 1000 == 0):
					if (hairstyle_tensor.shape[1] != 1) and (color_tensor.shape[1] != 1):
						img_tensor = torch.cat([hairstyle_tensor, color_tensor], dim = 3)
					elif hairstyle_tensor.shape[1] != 1:
						img_tensor = hairstyle_tensor
					elif color_tensor.shape[1] != 1:
						img_tensor = color_tensor
					else:
						img_tensor = None
					self.parse_and_log_images(x, x_hat, img_tensor, title='images_train', selected_description=selected_description)
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train', selected_description=selected_description)
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!', flush=True)
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			if batch_idx > 200:
				break

			w, hairstyle_text_inputs, color_text_inputs, selected_description_tuple, hairstyle_tensor, color_tensor = batch
			selected_description = ''
			for item in selected_description_tuple:
				selected_description+=item

			with torch.no_grad():
				w = w.to(self.device).float()
				hairstyle_text_inputs = hairstyle_text_inputs.to(self.device)
				color_text_inputs = color_text_inputs.to(self.device)
				hairstyle_tensor = hairstyle_tensor.to(self.device)
				color_tensor = color_tensor.to(self.device)
				x, _ = self.net.decoder([w], input_is_latent=True, randomize_noise=True, truncation=1)
				if hairstyle_tensor.shape[1] != 1:
					hairstyle_tensor_hairmasked = hairstyle_tensor * self.average_color_loss.gen_hair_mask(hairstyle_tensor)
				else:
					hairstyle_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
				if color_tensor.shape[1] != 1:
					color_tensor_hairmasked = color_tensor * self.average_color_loss.gen_hair_mask(color_tensor)
				else:
					color_tensor_hairmasked = torch.Tensor([0]).unsqueeze(0).cuda()
				w_hat = w + 0.1 * self.net.mapper(w, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor_hairmasked, color_tensor_hairmasked)
				x_hat, _ = self.net.decoder([w_hat], input_is_latent=True, randomize_noise=True, truncation=1)
				loss, cur_loss_dict = self.calc_loss(w, x, w_hat, x_hat, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor, color_tensor, selected_description)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			if (hairstyle_tensor.shape[1] != 1) and (color_tensor.shape[1] != 1):
				img_tensor = torch.cat([hairstyle_tensor, color_tensor], dim = 3)
			elif hairstyle_tensor.shape[1] != 1:
				img_tensor = hairstyle_tensor
			elif color_tensor.shape[1] != 1:
				img_tensor = color_tensor
			else:
				img_tensor = None
			self.parse_and_log_images(x, x_hat, img_tensor, title='images_val', selected_description=selected_description, index=batch_idx)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test', selected_description=selected_description)

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'latest_model.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		params = list(self.net.mapper.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.latents_train_path:
			train_latents = torch.load(self.opts.latents_train_path)
		else: 
			train_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda()
			train_latents = []
			for b in range(self.opts.train_dataset_size // self.opts.batch_size):
				with torch.no_grad():
					_, train_latents_b = self.net.decoder([train_latents_z[b: b + self.opts.batch_size]],
														  truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True)
					train_latents.append(train_latents_b)
			train_latents = torch.cat(train_latents)

		if self.opts.latents_test_path:
			test_latents = torch.load(self.opts.latents_test_path)
		else:
			test_latents_z = torch.randn(self.opts.train_dataset_size, 512).cuda()
			test_latents = []
			for b in range(self.opts.test_dataset_size // self.opts.test_batch_size):
				with torch.no_grad():
					_, test_latents_b = self.net.decoder([test_latents_z[b: b + self.opts.test_batch_size]],
													  truncation=0.7, truncation_latent=self.net.latent_avg, return_latents=True)
					test_latents.append(test_latents_b)
			test_latents = torch.cat(test_latents)

		train_dataset_celeba = LatentsDataset(latents=train_latents.cpu(),
		                                      opts=self.opts,
		                                      status='train')
		test_dataset_celeba = LatentsDataset(latents=test_latents.cpu(),
		                                      opts=self.opts,
		                                      status='test')
		train_dataset = train_dataset_celeba
		test_dataset = test_dataset_celeba
		print("Number of training samples: {}".format(len(train_dataset)), flush=True)
		print("Number of test samples: {}".format(len(test_dataset)), flush=True)
		return train_dataset, test_dataset

	def calc_loss(self, w, x, w_hat, x_hat, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor, color_tensor, selected_description):
		loss_dict = {}
		loss = 0.0
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement = self.id_loss(x_hat, x)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda * self.opts.attribute_preservation_lambda

		if self.opts.text_manipulation_lambda > 0:
			if hairstyle_text_inputs.shape[1] != 1:
				loss_text_hairstyle = self.clip_loss(x_hat, hairstyle_text_inputs).mean()
				loss_dict['loss_text_hairstyle'] = float(loss_text_hairstyle)
				loss += loss_text_hairstyle * self.opts.text_manipulation_lambda
			if color_text_inputs.shape[1] != 1:
				loss_text_color = self.clip_loss(x_hat, color_text_inputs).mean()
				loss_dict['loss_text_color'] = float(loss_text_color)
				loss += loss_text_color * self.opts.text_manipulation_lambda

		if self.opts.image_hairstyle_lambda > 0:
			if hairstyle_tensor.shape[1] != 1:
				if 'hairstyle_out_domain_ref' in selected_description:
					loss_img_hairstyle = self.image_embedding_loss((x_hat * self.average_color_loss.gen_hair_mask(x_hat)), (hairstyle_tensor * self.average_color_loss.gen_hair_mask(hairstyle_tensor))).mean()
					loss_dict['loss_img_hairstyle'] = float(loss_img_hairstyle)
					loss += loss_img_hairstyle * self.opts.image_hairstyle_lambda * self.opts.image_manipulation_lambda
			
		if self.opts.image_color_lambda > 0:
			if color_tensor.shape[1] != 1:
				loss_img_color = self.average_color_loss(color_tensor, x_hat)
				loss_dict['loss_img_color'] = float(loss_img_color)
				loss += loss_img_color * self.opts.image_color_lambda * self.opts.image_manipulation_lambda
				
		if self.opts.maintain_color_lambda > 0:
			if ((hairstyle_tensor.shape[1] != 1) or (hairstyle_text_inputs.shape[1] != 1)) and (color_tensor.shape[1] == 1) and (color_text_inputs.shape[1] == 1):
				loss_maintain_color_for_hairstyle = self.maintain_color_for_hairstyle_loss(x, x_hat)
				loss_dict['loss_maintain_color_for_hairstyle'] = float(loss_maintain_color_for_hairstyle)
				loss += loss_maintain_color_for_hairstyle * self.opts.maintain_color_lambda * self.opts.attribute_preservation_lambda
		if self.opts.background_lambda > 0:
			loss_background = self.background_loss(x, x_hat)
			loss_dict['loss_background'] = float(loss_background)
			loss += loss_background * self.opts.background_lambda * self.opts.attribute_preservation_lambda
		if self.opts.latent_l2_lambda > 0:
			loss_l2_latent = self.latent_l2_loss(w_hat, w)
			loss_dict['loss_l2_latent'] = float(loss_l2_latent)
			loss += loss_l2_latent * self.opts.latent_l2_lambda * self.opts.attribute_preservation_lambda
		loss_dict['loss'] = float(loss)
		return loss, loss_dict

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix, selected_description):
		if prefix == 'train':
			print('Metrics for {}, step {}'.format(prefix, self.global_step), selected_description, flush=True)
		else:
			print('Metrics for {}, step {}'.format(prefix, self.global_step), flush=True)
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value, flush=True)

	def parse_and_log_images(self, x, x_hat, img_tensor, title, selected_description, index=None):
		if index is None:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}-{selected_description}.jpg')
		else:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}-{str(index).zfill(5)}-{selected_description}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		if img_tensor is not None:
			if img_tensor.shape[3] == 1024:
				torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu(), img_tensor.detach().cpu()]), path,
									 normalize=True, scale_each=True, range=(-1, 1), nrow=3)
			elif img_tensor.shape[3] == 2048:
				torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu(), img_tensor[:,:,:,0:1024].detach().cpu(), img_tensor[:,:,:,1024::].detach().cpu()]), path,
									 normalize=True, scale_each=True, range=(-1, 1), nrow=4)				
		else:
			torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu()]), path,
									 normalize=True, scale_each=True, range=(-1, 1), nrow=2)				

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict