import os
import cv2
import json
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from PIL import Image
import glob
import sys
from scipy import signal
import random
import soundfile as sf

class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        data = []

        # to make it such that it works for items outside of the test stimuli
        # print("line before loop")
        for item in os.listdir(args.data_path):
            # print("entered loop")
            if os.path.splitext(item)[1] == '.wav': # all audios must be in .wav format
                data.append(item)
                data = sorted(data, key=lambda x: int(x[5:-4]))
        print(data)
        print(len(data))

        self.audio_path = args.data_path 
        self.mode = mode
        self.transforms = transforms

        # initialize audio transform
        self._init_atransform()

        #  Retrieve list of audio and video files
        self.video_files = []
        
        for item in data:
            self.video_files.append(item)
        print('# of audio files = %d ' % len(self.video_files))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.video_files)  

    def __getitem__(self, idx):
        print(f"getting item {idx}")
        wav_file = self.video_files[idx]

        try:
            # Audio
            samples, samplerate = sf.read(os.path.join(self.audio_path,wav_file))
        except Exception as e:
            print(f"Error loading {self.audio_path + wav_file}: {e}")
            raise e

        # repeat in case audio is too short
        resamples = np.tile(samples,10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512,noverlap=353)
        spectrogram = np.log(spectrogram+ 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram-mean,std+1e-9)
        
        return spectrogram, resamples, wav_file


