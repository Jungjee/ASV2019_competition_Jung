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
			nfft = 2048,
			mode = 'psd')
		_, _, spec2 = spectrogram(x = wav,
			fs = 16000,
			window = 'hamming',
			nperseg = int(16000*0.001*50),
			noverlap = int(16000*0.001*30),
			nfft = 2048,
			mode = 'phase')
		_, _, spec3 = spectrogram(x = wav,
			fs = 16000,
			window = 'hamming',
			nperseg = int(16000*0.001*50),
			noverlap = int(16000*0.001*30),
			nfft = 2048,
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

#eval
DB_dir = 'C:/DB/ASVspoof2019/PA/ASVspoof2019_PA_eval/'
scp_dir = 'C:/DB/ASVspoof2019/PA/feature/spectrogram/'
nb_proc = 16
dataset = 'eval'


if __name__ == '__main__':
	list_f_dir = []
	for r, ds, fs in os.walk(DB_dir):
		for f in fs:
			if f.split('.')[-1] != 'flac':
				continue
			fn = '/'.join([r, f]).replace('\\', '/')
			key = fn.split('/')[-1].split('.')[0]
			print(fn, key)
			list_f_dir.append('%s %s\n'%(key, fn))
	print('='*5 + 'done' + '='*5)
	print(len(list_f_dir))


	print('Processing Eval set')
	if not os.path.exists(scp_dir[:-1] + '_%s/'%dataset):
		os.makedirs(scp_dir[:-1] + '_%s/'%dataset)
	list_proc = []
	nb_utt_per_proc = int(len(list_f_dir) / nb_proc)
	for i in range(nb_proc):
		if i == nb_proc - 1:
			lines = list_f_dir[i * nb_utt_per_proc : ]
		else:
			lines = list_f_dir[i * nb_utt_per_proc : (i+1) * nb_utt_per_proc]

		print(len(lines))
		list_proc.append(Process(target = extract_spectrograms, args = (lines, scp_dir[:-1] +'_%s/wav_%d'%(dataset, i))))
		print('%d'%i)

	for i in range(nb_proc):
		list_proc[i].start()
		print('start %d'%i)
	for i in range(nb_proc):
		list_proc[i].join()

	join_scp(f_dir = scp_dir[:-1] + '_%s/wav'%(dataset), nb_proc = nb_proc)










