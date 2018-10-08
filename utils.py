import os
import glob
import numpy as np
from PIL import Image
import tqdm as tq

class ImageUtils():
	def __init__(self):
		self.x_rays_test = []
		self.x_rays_train = []
		self.curr_test_batch = 0
		self.curr_train_batch = 0

		self.train_dir = 'train'
		self.test_dir = 'test'
		self.bacteria = 'PNEUMONIA'
		self.normal = 'NORMAL'
		self.load_test_set()
		self.load_train_set()

	def load_test_set(self):
		for img in tq.tqdm(glob.glob(self.test_dir + '/' + self.bacteria + '/*.jpeg')):
			self.x_rays_test.append({'image': img, 'class': [1, 0]})

		for img in tq.tqdm(glob.glob(self.test_dir + '/' + self.normal + '/*.jpeg')):
			self.x_rays_test.append({'image': img, 'class': [0, 1]})
		np.random.shuffle(self.x_rays_test)

	def load_train_set(self):
		for img in tq.tqdm(glob.glob(self.train_dir + '/' + self.bacteria + '/*.jpeg')):
			self.x_rays_train.append({'image': img, 'class': [1, 0]})

		for img in tq.tqdm(glob.glob(self.train_dir + '/' + self.normal + '/*.jpeg')):
			self.x_rays_train.append({'image': img, 'class': [0, 1]})
		np.random.shuffle(self.x_rays_train)

	def get_test_batch(self, size):
		start_idx = self.curr_test_batch * size
		end_idx = start_idx + size
		self.curr_test_batch += 1
		if(end_idx >= len(self.x_rays_test)):
			self.curr_test_batch = 0
			end_idx = len(self.x_rays_test)-1
		rtrn_array = {'image':[], 'class':[]}
		for i in tq.tqdm(range(start_idx, end_idx)):
			img_class = self.x_rays_test[i]['class']
			img = Image.open(self.x_rays_test[i]['image']).resize((500, 500), Image.ANTIALIAS)
			img = np.asarray(img)
			if(len(img.shape) == 2):
				rtrn_array['image'].append(img.reshape(500, 500, 1))
				rtrn_array['class'].append(img_class)
		return rtrn_array


	def get_train_batch(self, size):
		start_idx = self.curr_train_batch * size
		end_idx = start_idx + size		
		self.curr_train_batch += 1
		if(end_idx >= len(self.x_rays_train)):
			self.curr_train_batch = 0
			end_idx = len(self.x_rays_train)-1
		rtrn_array = {'image':[], 'class':[]}
		for i in tq.tqdm(range(start_idx, end_idx)):
			img_class = self.x_rays_train[i]['class']
			img = Image.open(self.x_rays_train[i]['image']).resize((500, 500), Image.ANTIALIAS)
			img = np.asarray(img)
			if(len(img.shape) == 2):
				rtrn_array['image'].append(img.reshape(500, 500, 1))
				rtrn_array['class'].append(img_class)
		return rtrn_array
