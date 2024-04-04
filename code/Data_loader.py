import glob, numpy, os, random, soundfile, torch ,itertools
import numpy as np
import librosa
from torch.utils.data import Dataset
from typing import Tuple
from scipy import signal
import  torchaudio.transforms as AudioT
from torch import nn


class train_loader(object):
	def __init__(self, train_list, train_path, num_frames):
		self.train_path = train_path #train 파일 경로
		self.num_frames = num_frames #프레임수
		self.MFCC = AudioT.MFCC(sample_rate=16000,n_mfcc=80,log_mels=True,melkwargs=
         {'n_fft': 512, 'hop_length':160,'win_length':400,'n_mels':80, "center": False})
		self.Norm = nn.InstanceNorm1d(80)
		# Load and configure augmentation files
		"""self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))"""
		# Load data & labels
		self.data_list  = [] 
		self.data_label = []
		lines = open(train_list).read().splitlines()
		#print(lines)
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		#print(dictkeys)
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)
		#print(self.data_label)
		#print(self.data_list)			
	def loadWAV(self,filename, max_frames):

        # Maximum audio length
		max_audio = 300 * 160 + 240

        # Read wav file and convert to torch tensor
		audio, sample_rate = soundfile.read(filename)

		audiosize = audio.shape[0]

		if audiosize <= max_audio: #오디오길이 padding
			shortage = max_audio - audiosize + 1
			audio = numpy.pad(audio, (0, shortage), 'wrap')
			audiosize = audio.shape[0]

		startframe = np.int64(random.random() * (audio.shape[0] - max_audio))

		audio = audio[startframe:startframe + max_audio]
		audio = np.stack([audio], axis=0).astype(numpy.float32)
		return audio
	
	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		data = self.loadWAV(self.data_list[index],  max_frames=self.num_frames)
		data = self.MFCC(torch.from_numpy(data))
		data = self.Norm(data).reshape(1, 80, -1)
		#audio = numpy.stack([audio],axis=0)
		# Data Augmentation
		"""augtype = random.randint(0,5)
		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1: # Reverberation
			audio = self.add_rev(audio)
		elif augtype == 2: # Babble
			audio = self.add_noise(audio, 'speech')
		elif augtype == 3: # Music
			audio = self.add_noise(audio, 'music')
		elif augtype == 4: # Noise
			audio = self.add_noise(audio, 'noise')
		elif augtype == 5: # Television noise
			audio = self.add_noise(audio, 'speech')
			audio = self.add_noise(audio, 'music')"""
		return data, self.data_label[index]

	def __len__(self):
		return len(self.data_list)

	def add_rev(self, audio):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:self.num_frames * 160 + 240]

	def add_noise(self, audio, noisecat):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4) 
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			length = self.num_frames * 160 + 240
			if noiseaudio.shape[0] <= length:
				shortage = length - noiseaudio.shape[0]
				noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
			start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
			noiseaudio = noiseaudio[start_frame:start_frame + length]
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio
	


class test_loader(Dataset):
	def __init__(self, test_list, test_path, eval_frames, num_eval):
		self.test_path = test_path 
		self.num_eval = num_eval
		self.max_frames =eval_frames
		self.Norm = nn.InstanceNorm1d(80)
		# Load data & labels
		self.data_list  = [] 
		self.data_label = []
		self.MFCC = AudioT.MFCC(sample_rate=16000, n_mfcc=80, log_mels=True, melkwargs=
        {'n_fft': 512, 'hop_length': 160, 'win_length': 400, 'n_mels': 80, "center": False})
		with open(test_list) as f:
			lines = f.readlines()
		
		files = list(itertools.chain(*[x.strip().split()[-2:] for x in lines]))
		set_files = list(set(files))
		set_files.sort()
		#print(set_files)

		for index, line in enumerate(set_files):
			file_name = os.path.join(test_path, line)
			self.data_list.append(file_name)

	def loadWAV(self, filename):

        # Maximum audio length
		audio_length =  300 * 160 + 240

        # Read wav file and convert to torch tensor
		data, sr = librosa.load(filename, sr=16000)

		if len(data) > audio_length:
			offset = len(data) - audio_length
			#offset = np.random.randint(max_offset)
			data = data[offset:(audio_length + offset)]

		else:
			if audio_length > len(data):
				offset = audio_length - len(data)
			else:
				offset = 0
			data = np.pad(data, (offset, audio_length - len(data) - offset), "constant")

		return data



	def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
		# Read the utterance and randomly select the segment
		data = self.loadWAV(self.data_list[index])
		data = self.MFCC(torch.from_numpy(data))
		data = self.Norm(data).reshape(1, 80, -1)
		return data, self.data_list[index]

	def __len__(self):
		return len(self.data_list)
