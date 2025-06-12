import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import pickle, pandas as pd
import numpy
import json
import yaml

meld_speakers = ['Rachel', 'Joey', 'Ross', 'Monica', 'Chandler', 'Phoebe']
# meld_speakers = ['Rachel', 'Joey', 'Monica']#, 'Ross', 'Chandler', 'Phoebe']
# meld_speakers = ['Rachel', 'Joey', 'Chandler']#, 'Ross', 'Monica', 'Phoebe']
iemocap_speakers = ['Ses01_M', 'Ses01_F', 'Ses02_M', 'Ses02_F', 'Ses03_M', 'Ses03_F', 'Ses04_M', 'Ses04_F']
# iemocap_speakers = ['Ses02_M', 'Ses02_F', 'Ses04_M', 'Ses04_F']
no_speakers = ['Ross', 'Monica', 'Phoebe']
# iemocap_speakers = ['M', 'F']

class IEMOCAPDataset(Dataset):

	def __init__(self, train=True):
		self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
		self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
		self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features_raw.pkl', 'rb'), encoding='latin1')
		#self.testVid = pickle.load(open('./IEMOCAP_features/IEMOCAP_features.pkl', 'rb'), encoding='latin1')

		_, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4,\
		_, _, _, _ = pickle.load(open('./IEMOCAP_features/iemocap_features_roberta.pkl', 'rb'), encoding='latin1')
		'''
		label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
		'''
		self.keys = [x for x in (self.trainVid if train else self.testVid)]
		self.speakerNames = {}
		# self.keys = []
		# tag = 0
		for diaid in self.videoSpeakers.keys():
			# if diaid[:5] == 'Ses01' or diaid[:5] == 'Ses05':
			# 	if train:
			# 		self.keys.append(diaid)
			# elif diaid[:5] == 'Ses03':
			# 	if not train:
			# 		self.keys.append(diaid)
			# else:
			# 	if train == tag & 1:
			# 		self.keys.append(diaid)
			# 	tag += 1
			self.speakerNames[diaid] = []
			for sp in self.videoSpeakers[diaid]:
				speaker = diaid[:5] + '_' + sp
				if speaker in iemocap_speakers:
					self.speakerNames[diaid].append(iemocap_speakers.index(speaker))
				else:
					self.speakerNames[diaid].append(-1)
		
		self.len = len(self.keys)

	def __getitem__(self, index):
		vid = self.keys[index]
		return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
			   torch.FloatTensor(numpy.array(self.roberta2[vid])),\
			   torch.FloatTensor(numpy.array(self.roberta3[vid])),\
			   torch.FloatTensor(numpy.array(self.roberta4[vid])),\
			   torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
			   torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
			   torch.FloatTensor(numpy.array([[1,0] if x=='M' else [0,1] for x in\
								  self.videoSpeakers[vid]])),\
			   torch.FloatTensor(numpy.array([1 if i != -1 else 0 for i in self.speakerNames[vid]])),\
			   torch.FloatTensor(numpy.array([1] * len(self.speakerNames[vid]))),\
			   torch.LongTensor(numpy.array(self.videoLabels[vid])),\
			   torch.LongTensor(numpy.array(self.speakerNames[vid])),\
			   vid

	def __len__(self):
		return self.len

	def collate_fn(self, data):
		dat = pd.DataFrame(data)
		return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<10 else pad_sequence(dat[i], True, -1) if i<11 else dat[i].tolist() for i in dat]

class MELDDataset(Dataset):

	def __init__(self, path, train=True):
		self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText,\
		self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid,\
		self.testVid, _ = pickle.load(open(path, 'rb'))
		self.keys = [x for x in (self.trainVid if train else self.testVid)]
		self.trainvis = pickle.load(open('./V/meld_train_vision_utt.pkl', 'rb'))['train']
		self.valvis = pickle.load(open('./V/meld_val_vision_utt.pkl', 'rb'))['val']
		self.testvis = pickle.load(open('./V/meld_test_vision_utt.pkl', 'rb'))['test']
		self.visdata = {}
		self.vismask = {}
		self.speakerNames = {}
		with open('MELD/utterance-ordered.json', 'r') as fl:
			self.utterance_ordered = json.load(fl)
		self.train_err = 'dia125_utt3'
		self.test_err = 'dia220_utt0'

		i, j = 0, 0
		if train:
			nkeys = []
			diaids = list(self.utterance_ordered['train'].keys())
			for diaid in diaids:
				flag = 1
				self.speakerNames[self.keys[i]], self.visdata[self.keys[i]], self.vismask[self.keys[i]] = [], [], []
				for uttid in self.utterance_ordered['train'][diaid]:
					self.visdata[self.keys[i]].append(numpy.pad(self.trainvis['vision'][j], pad_width=((0, 2), (0, 0)), mode='constant'))
					self.vismask[self.keys[i]].append(numpy.pad(self.trainvis['vision_utt_mask'][j], pad_width=(0, 2), mode='constant'))
					if uttid != self.train_err:
						j += 1
					with open('MELD/raw-texts/train/' + uttid + '.json', 'r') as fl:
						sp = json.load(fl)['Speaker']
						if sp in meld_speakers:
							self.speakerNames[self.keys[i]].append(meld_speakers.index(sp))
						else:
							self.speakerNames[self.keys[i]].append(-1)
						if sp in no_speakers:
							flag = 0
				if flag:
					nkeys.append(self.keys[i])
				i += 1
			j = 0
			diaids = list(self.utterance_ordered['val'].keys())
			for diaid in diaids:
				flag = 1
				self.speakerNames[self.keys[i]], self.visdata[self.keys[i]], self.vismask[self.keys[i]] = [], [], []
				for uttid in self.utterance_ordered['val'][diaid]:
					self.visdata[self.keys[i]].append(self.valvis['vision'][j])
					self.vismask[self.keys[i]].append(self.valvis['vision_utt_mask'][j])
					j += 1
					with open('MELD/raw-texts/val/' + uttid + '.json', 'r') as fl:
						sp = json.load(fl)['Speaker']
						if sp in meld_speakers:
							self.speakerNames[self.keys[i]].append(meld_speakers.index(sp))
						else:
							self.speakerNames[self.keys[i]].append(-1)
						if sp in no_speakers:
							flag = 0
				if flag:
					nkeys.append(self.keys[i])
				i += 1
			# self.keys = nkeys
		else:
			diaids = list(self.utterance_ordered['test'].keys())
			for diaid in diaids:
				self.speakerNames[self.keys[i]], self.visdata[self.keys[i]], self.vismask[self.keys[i]] = [], [], []
				for uttid in self.utterance_ordered['test'][diaid]:
					self.visdata[self.keys[i]].append(self.testvis['vision'][j])
					self.vismask[self.keys[i]].append(self.testvis['vision_utt_mask'][j])
					if uttid != self.test_err:
						j += 1
					with open('MELD/raw-texts/test/' + uttid + '.json', 'r') as fl:
						sp = json.load(fl)['Speaker']
						if sp in meld_speakers:
							self.speakerNames[self.keys[i]].append(meld_speakers.index(sp))
						else:
							self.speakerNames[self.keys[i]].append(-1)
				i += 1
		self.len = len(self.keys)
		cnt1, cnt2 = 0, 0
		for k in self.keys:
			for sp in self.speakerNames[k]:
				if sp == -1:
					cnt2 += 1
				else:
					cnt1 += 1
		print(train, self.len, cnt1, cnt2)
		_, _, _, self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
			_, self.trainIds, self.testIds, self.validIds \
			= pickle.load(open("./MELD_features/meld_features_roberta.pkl", 'rb'), encoding='latin1')

	def __getitem__(self, index):
		vid = self.keys[index]
		return torch.FloatTensor(numpy.array(self.roberta1[vid])),\
			   torch.FloatTensor(numpy.array(self.roberta2[vid])),\
			   torch.FloatTensor(numpy.array(self.roberta3[vid])),\
			   torch.FloatTensor(numpy.array(self.roberta4[vid])),\
			   torch.FloatTensor(numpy.array(self.videoVisual[vid])),\
			   torch.FloatTensor(numpy.array(self.videoAudio[vid])),\
			   torch.FloatTensor(numpy.array(self.videoSpeakers[vid])),\
			   torch.FloatTensor(numpy.array(self.visdata[vid])),\
			   torch.FloatTensor(numpy.array(self.vismask[vid])),\
			   torch.FloatTensor(numpy.array([1 if i != -1 else 0 for i in self.speakerNames[vid]])),\
			   torch.FloatTensor(numpy.array([1] * len(self.speakerNames[vid]))),\
			   torch.LongTensor(numpy.array(self.videoLabels[vid])),\
			   torch.LongTensor(numpy.array(self.speakerNames[vid])),\
			   vid  

	def __len__(self):
		return self.len

	def return_labels(self):
		return_label = []
		for key in self.keys:
			return_label+=self.videoLabels[key]
		return return_label

	def collate_fn(self, data):
		dat = pd.DataFrame(data)
		return [pad_sequence(dat[i]) if i<9 else pad_sequence(dat[i], True) if i<12 else pad_sequence(dat[i], True, -1) if i<13 else dat[i].tolist() for i in dat]
