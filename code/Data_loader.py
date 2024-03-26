import glob, numpy, os, random, soundfile, torch
import librosa
from scipy import signal

class train_loader(object):
	def __init__(self, train_list, train_path, num_frames):
		self.train_path = train_path #train 파일 경로
		self.num_frames = num_frames #프레임수
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
		
		dictkeys = list(set([x.split()[0] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }

		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name = os.path.join(train_path, line.split()[1])
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)
			
		self.pad2d = lambda a, i: a[:, 0:i] if a.shape[1] > i else numpy.hstack((a, numpy.zeros((a.shape[0], i-a.shape[1]))))


	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		audio, sr = librosa.load(self.data_list[index], sr=44100)
		mfcc = librosa.feature.mfcc(audio=audio, sr=sr, n_mfcc=13, n_fft=int(0.02 * sr), hop_length=int(0.01 * sr))
		padded_mfcc = self.pad2d(mfcc, 40)
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
		return torch.FloatTensor(padded_mfcc), self.data_label[index]

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