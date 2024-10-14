import numpy as np
import librosa


class AudioAugmentation:
    def __init__(self):
        '''
        '''


    def LoadAudioData(self, audio_path):
        self.data, self.sr = librosa.load(audio_path[6])
        return self.data, self.sr
    
        
    def add_noise(self,data,random = False):
        # parameter
        rate = 0.035
        threshold = 0.075

        if random:
            rate=np.random.random()*threshold

        noise=rate*np.random.uniform()*np.amax(data)
        augmented_data=data+noise*np.random.normal(size=data.shape[0])

        return augmented_data

    def shifting(self,data):
        # parameter
        rate = 1000

        augmented_data=int(np.random.uniform(low=-5,high=5)*rate)
        augmented_data=np.roll(data,augmented_data)

        return augmented_data

    def pitching(self,data,sr,random = False):
        # parameter
        pitch_factor = 0.7

        if random:
            pitch_factor=np.random.random() * pitch_factor

        return librosa.effects.pitch_shift(data,sr=sr,n_steps=pitch_factor)

    def streching(self,data,rate=0.8):
        return librosa.effects.time_stretch(data,rate=rate)
        

    def GenerateAudioAugmentation(self, data, sr):

        ## noised Audio
        self.noised_audio=self.add_noise(data)

        ## streched audio
        self.stretched_audio=self.streching(data)

        ## shifted Audio
        self.shifted_audio=self.shifting(data)

        ## Piched Audio
        self.pitched_audio=self.pitching(data,sr)

        return self.noised_audio, self.stretched_audio, self.shifted_audio, self.pitched_audio
