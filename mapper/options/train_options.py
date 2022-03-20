from argparse import ArgumentParser


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--no_coarse_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_medium_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_fine_mapper', default=False, action="store_true")
		self.parser.add_argument('--latents_train_path', default="", type=str, help="The latents for the training")
		self.parser.add_argument('--latents_test_path', default="", type=str, help="The latents for the validation")

		self.parser.add_argument('--hairstyle_ref_img_train_path', default="", type=str, help="The hairstyle reference image for the training")
		self.parser.add_argument('--hairstyle_ref_img_test_path', default="", type=str, help="The hairstyle reference image for the validation")
		self.parser.add_argument('--color_ref_img_train_path', default="", type=str, help="The color reference image for the training")
		self.parser.add_argument('--color_ref_img_test_path', default="", type=str, help="The color reference image for the validation")
		self.parser.add_argument('--color_ref_img_in_domain_path', default="", type=str, help="The color reference image in domain for the augmentation")
		self.parser.add_argument('--num_for_each_augmented_color', default=4000, type=int, help='Number for each augmented color')

		self.parser.add_argument('--hairstyle_manipulation_prob', default=0.5, type=float, help='Probability of only manipulating the hairstyle')
		self.parser.add_argument('--color_manipulation_prob', default=0.2, type=float, help='Probability of only manipulating the color')
		self.parser.add_argument('--both_manipulation_prob', default=0.27, type=float, help='Probability of simultaneously manipulating hairstyle and color')

		self.parser.add_argument('--hairstyle_text_manipulation_prob', default=0.5, type=float, help='Probability of using text to manipulate hairstyle')
		self.parser.add_argument('--color_text_manipulation_prob', default=0.5, type=float, help='Probability of using text to manipulate color')
		self.parser.add_argument('--color_in_domain_ref_manipulation_prob', default=0.25, type=float, help='Probability of using in-domain reference image to manipulate color')

		self.parser.add_argument('--train_dataset_size', default=5000, type=int, help="Will be used only if no latents are given")
		self.parser.add_argument('--test_dataset_size', default=1000, type=int, help="Will be used only if no latents are given")

		self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.0005, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')


		self.parser.add_argument('--text_manipulation_lambda', default=2.0, type=float, help='Text manipulation loss multiplier factor')
		self.parser.add_argument('--image_manipulation_lambda', default=1.0, type=float, help='Image manipulation loss multiplier factor')
		self.parser.add_argument('--attribute_preservation_lambda', default=1.0, type=float, help='Attribute preservation loss multiplier factor')


		self.parser.add_argument('--image_hairstyle_lambda', default=5.0, type=float, help='Image-based hairstyle manipulation loss multiplier factor')
		self.parser.add_argument('--image_color_lambda', default=0.02, type=float, help='Image-based color manipulation loss multiplier factor')

		self.parser.add_argument('--id_lambda', default=0.3, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--maintain_color_lambda', default=0.02, type=float, help='Color retention loss multiplier factor')
		self.parser.add_argument('--background_lambda', default=1.0, type=float, help='Background loss multiplier factor')
		self.parser.add_argument('--latent_l2_lambda', default=0.8, type=float, help='Latent L2 loss multiplier factor')

		self.parser.add_argument('--parsenet_weights', default='../pretrained_models/parsenet.pth', type=str, help='Path to Parsing model weights')
		self.parser.add_argument('--stylegan_weights', default='../pretrained_models/stylegan2-ffhq-config-f.pt', type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_size', default=1024, type=int)
		self.parser.add_argument('--ir_se50_weights', default='../pretrained_models/model_ir_se50.pth', type=str, help="Path to facial recognition network used in ID loss")
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to HairCLIP model checkpoint')

		self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=2000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=2000, type=int, help='Model checkpoint interval')

		self.parser.add_argument('--hairstyle_description', required=True, type=str, help='Hairstyle text prompt list')
		self.parser.add_argument('--color_description', required=True, type=str, help='Color text prompt, eg: purple, red, orange')


	def parse(self):
		opts = self.parser.parse_args()
		return opts