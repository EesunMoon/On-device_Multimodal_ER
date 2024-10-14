import numpy as np
import pandas as np
import IPython
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display

from DataAnalyze import DataAnalyze


class DisplayEmotion:
    def __init__(self):
        
        self.colors={'disgust':'#804E2D',
                    'happy':'#F19C0E',
                    'sad':'#478FB8',
                    'neutral':'#4CB847',
                    'fear':'#7D55AA',
                    'angry':'#C00808',
                    'surprise':'#EE00FF'}


    def DisplayAudioEmotion(self, audio_path):
        ## print 
        print('Disgust Audio Sample\n')
        IPython.display.Audio(audio_path[0])
        
        print('Happy Audio Sample\n')
        IPython.display.Audio(audio_path[1])

        print('Sad Audio Sample\n')
        IPython.display.Audio(audio_path[2])

        print('Neutral Audio Sample\n')
        IPython.display.Audio(audio_path[3])

        print('Fear Audio Sample\n')
        IPython.display.Audio(audio_path[4])

        print('Angry Audio Sample\n')
        IPython.display.Audio(audio_path[5])

        print('Surprise Audio Sample\n')
        IPython.display.Audio(audio_path[6])

    
    def DisplayAudioAugmentation(self, audio_path, data, sr, noised_audio, stretched_audio, shifted_audio, pitched_audio):
        
        ## original Audio
        print('Original Audio')
        plt.figure(figsize=(12,5))
        librosa.display.waveshow(y=data,sr=sr,color='#EE00FF')
        IPython.display.Audio(audio_path[6])

        ## noised Audio
        print('Noised Audio')
        plt.figure(figsize=(12,5))
        librosa.display.waveshow(y=noised_audio,sr=sr,color='#EE00FF')
        IPython.display.Audio(noised_audio,rate=sr)

        ## streched audio
        print('Streched Audio')
        plt.figure(figsize=(12,5))
        librosa.display.waveshow(y=stretched_audio,sr=sr,color='#EE00FF')
        IPython.display.Audio(stretched_audio,rate=sr)

        ## shifted Audio
        print('Shifted Audio')
        plt.figure(figsize=(12,5))
        librosa.display.waveshow(y=shifted_audio,sr=sr,color='#EE00FF')
        IPython.display.Audio(shifted_audio,rate=sr)


        # Piched Audio
        print('Piched Audio')
        plt.figure(figsize=(12,5))
        librosa.display.waveshow(y=pitched_audio,sr=sr,color='#EE00FF')
        IPython.display.Audio(pitched_audio,rate=sr)
