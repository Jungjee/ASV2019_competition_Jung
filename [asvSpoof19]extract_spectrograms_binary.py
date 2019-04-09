# -*- coding: utf-8 -*-
from scipy.signal import spectrogram
import numpy as np
import struct
from multiprocessing import Process
import soundfile as sf
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap



def pre_emp(x):
	return np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32)


def extract_spectrograms(lines, dir_trg):
	
	
	f_scp = open(dir_trg + '.scp', 'w')
	f_ark = open(dir_trg + '.ark', 'wb')
	for line in lines:
		key, fn = line.strip().split(' ')
		
		wav ,_= sf.read(fn, dtype='int16')
		wav = pre_emp(wav)
		pointer = f_ark.tell()
		arkf_dir = os.path.abspath(dir_trg + '.ark').replace('\\', '/')
		arkf_dir = '/'.join(arkf_dir.split('/')[-4:])
		f_scp.write('%s %s %d\n'%(key, arkf_dir, pointer))

		_, _, spec1 = spectrogram(x = wav,
			fs = 16000,
			window = 'hamming',
			nperseg = int(16000*0.001*50),
			noverlap = int(16000*0.001*30),
			nfft = 1024,
			mode = 'psd')
		_, _, spec2 = spectrogram(x = wav,
			fs = 16000,
			window = 'hamming',
			nperseg = int(16000*0.001*50),
			noverlap = int(16000*0.001*30),
			nfft = 1024,
			mode = 'phase')
		_, _, spec3 = spectrogram(x = wav,
			fs = 16000,
			window = 'hamming',
			nperseg = int(16000*0.001*50),
			noverlap = int(16000*0.001*30),
			nfft = 1024,
			mode = 'magnitude')
                        
		spec = np.asarray([spec1, spec2, spec3], dtype = np.float32).transpose((2, 1, 0)).flatten()
		#print(spec.shape)
		#exit()
		f_ark.write(struct.pack('<L', spec.shape[0]))
		f_ark.write(struct.pack('<%df'%spec.shape[0], *spec.tolist()))
	
	f_scp.close()
	f_ark.close()
	

def join_scp(f_dir, nb_proc):
	f_scp = open(f_dir + '.scp', 'w')

	for i in range(nb_proc):
		with open(f_dir + '_%d.scp'%i, 'r') as f_read:
			lines = f_read.readlines()
		for line in lines:
			f_scp.write(line)

		os.remove(f_dir + '_%d.scp'%i)

	f_scp.close()
#======================================================================#
#======================================================================#

''' ##trn
DB_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_train/'
scp_dir = 'C:/DB/ASVspoof2019/PA/feature/spectrogram/'
meta_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_protocols_v1/ASVspoof2019.PA.cm.train.trn.txt'
nb_proc = 16
dataset = 'trn'
'''
'''dev
DB_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_dev/'
scp_dir = 'C:/DB/ASVspoof2019/PA/feature/spectrogram/'
meta_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_protocols_v1/ASVspoof2019.PA.cm.dev.trl.txt'
nb_proc = 16
dataset = 'dev'
'''
''' ##trn
DB_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_train/'
scp_dir = 'C:/DB/ASVspoof2019/PA/feature/spectrogram_nfft1024/'
meta_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_protocols_v1/ASVspoof2019.PA.cm.train.trn.txt'
nb_proc = 12
dataset = 'trn'
'''
#'''dev
DB_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_dev/'
scp_dir = 'C:/DB/ASVspoof2019/PA/feature/spectrogram_nfft1024/'
meta_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_protocols_v1/ASVspoof2019.PA.cm.dev.trl.txt'
nb_proc = 12
dataset = 'dev'
#'''

if __name__ == '__main__':
	meta_scp = open(meta_dir, 'r').readlines()
	d_meta = {}
	for l in meta_scp:
		_, key, _, _, t = l.strip().split(' ')
		#print(key, t)
		#exit()
		if t[0] == 'b':
			#key = key + '_1'
			d_meta[key] = 1
		elif t[0] == 's':
			#key = key + '_0'
			d_meta[key] = 0
		else:
			raise ValueError('!!')

	list_f_dir_b = []
	list_f_dir_s = []
	for r, ds, fs in os.walk(DB_dir):
		for f in fs:
			if f.split('.')[-1] != 'flac':
				continue
			fn = '/'.join([r, f]).replace('\\', '/')
			key = fn.split('/')[-1].split('.')[0]
			print(fn, key)

			if d_meta[key] == 1:
				list_f_dir_b.append('%s %s\n'%(key+'_1', fn))
			elif d_meta[key] == 0:
				list_f_dir_s.append('%s %s\n'%(key+'_0', fn))

	print('='*5 + 'done' + '='*5)
	print(len(list_f_dir_b), len(list_f_dir_b))


	print('Processing Bonafide set')
	if not os.path.exists(scp_dir[:-1] + '_%s/'%dataset):
		os.makedirs(scp_dir[:-1] + '_%s/'%dataset)
	list_proc = []
	nb_utt_per_proc = int(len(list_f_dir_b) / nb_proc)
	for i in range(nb_proc):
		if i == nb_proc - 1:
			lines = list_f_dir_b[i * nb_utt_per_proc : ]
		else:
			lines = list_f_dir_b[i * nb_utt_per_proc : (i+1) * nb_utt_per_proc]

		print(len(lines))
		list_proc.append(Process(target = extract_spectrograms, args = (lines, scp_dir[:-1] +'_%s/%s_wav_%d'%(dataset, 'bonafide', i))))
		print('%d'%i)

	for i in range(nb_proc):
		list_proc[i].start()
		print('start %d'%i)
	for i in range(nb_proc):
		list_proc[i].join()

	join_scp(f_dir = scp_dir[:-1] + '_%s/%s_wav'%(dataset, 'bonafide'), nb_proc = nb_proc)

	print('Processing Spoof set')
	if not os.path.exists(scp_dir[:-1] + '_%s/'%dataset):
		os.makedirs(scp_dir[:-1] + '_%s/'%dataset)
	list_proc = []
	nb_utt_per_proc = int(len(list_f_dir_s) / nb_proc)
	for i in range(nb_proc):
		if i == nb_proc - 1:
			lines = list_f_dir_s[i * nb_utt_per_proc : ]
		else:
			lines = list_f_dir_s[i * nb_utt_per_proc : (i+1) * nb_utt_per_proc]

		print(len(lines))
		list_proc.append(Process(target = extract_spectrograms, args = (lines, scp_dir[:-1] +'_%s/%s_wav_%d'%(dataset, 'spoof', i))))
		print('%d'%i)

	for i in range(nb_proc):
		list_proc[i].start()
		print('start %d'%i)
	for i in range(nb_proc):
		list_proc[i].join()

	join_scp(f_dir = scp_dir[:-1] + '_%s/%s_wav'%(dataset, 'spoof'), nb_proc = nb_proc)

	'''
	l_lines = []
	for r, ds, fs in os.walk(DB_dir):
		for f in fs:
			if f[-4:] != 'flac':
				continue
			l_lines.append('/'.join([r, f.replace('\\', '/')]))
			print(l_lines[-1])
	
	extract_spectrograms(
		l_lines,
		d_meta = d_meta,
		save_dir = scp_dir,
		fs = _fs,
		nfft = nfft,
		window = window,
		nperseg = int(_fs*0.001*nperseg),
		noverlap = int(_fs*0.001*noverlap),
		mode = mode)
	#def extract_spectrograms(lines, d_meta = None, save_dir = None, fs = 16000, window = 'hamming',
	#		nperseg = None, noverlap = None, nfft = None, scaling = None):
	'''









