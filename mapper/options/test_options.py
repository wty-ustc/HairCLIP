from argparse import ArgumentParser
class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--editing_type', type=str, help='Edit hairstyle or color or both, eg: both')
		self.parser.add_argument('--input_type', type=str, help='Input type is text or image, eg: text_image represents editing hairstyle with text and editing hair color with reference image')
		self.parser.add_argument('--hairstyle_description', default="", type=str, help='Hairstyle text prompt list')
		self.parser.add_argument('--color_description', default="", type=str, help='Color text prompt, eg: purple, red, orange')
		self.parser.add_argument('--hairstyle_ref_img_test_path', default="", type=str, help="The hairstyle reference image for the test")
		self.parser.add_argument('--color_ref_img_test_path', default="", type=str, help="The color reference image for the test")
		self.parser.add_argument('--num_of_ref_img', default=5, type=int, help='Number of reference images used for each target')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint')

		self.parser.add_argument('--no_coarse_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_medium_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_fine_mapper', default=False, action="store_true")
		self.parser.add_argument('--stylegan_size', default=1024, type=int)


		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--latents_test_path', default=None, type=str, help="The latents for the validation")
		self.parser.add_argument('--test_workers', default=0, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--start_index', default=0, type=int, help='Start index of test latents')
		self.parser.add_argument('--end_index', default=100, type=int, help='End index of test latents')


	def parse(self):
		opts = self.parser.parse_args()
		return opts