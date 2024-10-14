import numpy as np
import librosa
import pandas as pd


from AudioAugmentation import AudioAugmentation

class FeatureExtraction:
    def __init__(self):
        '''
        
        '''
        self.processed_data_path='/home/eesun/terser/data/audio/processed_Data/processed_data.csv'
        self.result=np.array([])
        

    def zcr(self,data,frame_length,hop_length):
        """
            zcr() : zero_crossing_rate - 음성신호가 0을 지나는 점
                                        즉, 음성 신호의 부호가 바뀌는 지점
        """
        zcr=librosa.feature.zero_crossing_rate(y=data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(zcr)
    
    def rmse(self,data,frame_length=2048,hop_length=512):
        """
            rms : 이상치 영향력 감소
        """
        # parameter
        rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(rmse)

    def mfcc(self,data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
        """
            mfcc : 기본 주파수 특성 추출
        """
        mfcc=librosa.feature.mfcc(y=data,sr=sr)
        return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

    def extract_features(self,data,sr):
        """
            extract_features() : 하나의 feature에
                    zcr || rmse || mfcc 로 stacked
        
        """
        # parameter
        frame_length = 2048
        hop_length = 512
        self.result=np.array([])

        self.result=np.hstack((self.result,
                        self.zcr(data,frame_length,hop_length),
                        self.rmse(data,frame_length=frame_length,hop_length=hop_length),
                        self.mfcc(data,sr,frame_length,hop_length)
                        ))
        
        return self.result

    def get_features(self, path):
        """
            get_features() : feature augmentation 수행 후 feature extraction (zcr / rms / mfcc) 
                1. origial audio
                    parameter
                        duration = 2.5
                        offset = 0,6
                2. noised audio
                    parameter
                        rate = 0.035
                        threshold = 0.075
                        if random:
                            rate=np.random.random()*threshold
                3. pitched audio
                    parameter
                        pitch_factor = 0.7
                        if random:
                            pitch_factor=np.random.random() * pitch_factor
                        n_steps = pitch_factor
                4. pitched + noised
        """
        # parameter
        aa = AudioAugmentation()
        duration = 2.5      # only load up to this much audio
        offset = 0.6        # start reading after this time

        data,sr=librosa.load(path,duration=duration,offset=offset)
        aud=self.extract_features(data,sr)
        self.audio=np.array(aud)
            
        self.noised_audio=aa.add_noise(data,random=True)
        aud2=self.extract_features(self.noised_audio,sr)
        self.audio=np.vstack((self.audio,aud2))
            
        self.pitched_audio=aa.pitching(data,sr,random=True)
        aud3=self.extract_features(self.pitched_audio,sr)
        self.audio=np.vstack((self.audio,aud3))
            
        self.pitched_audio1=aa.pitching(data,sr,random=True)
        pitched_noised_audio=aa.add_noise(self.pitched_audio1,random=True)
        aud4=self.extract_features(pitched_noised_audio,sr)
        self.audio=np.vstack((self.audio,aud4))
            
        return self.audio

    def SaveProcessedData(self, main_df):
        self.processed_data_path='/home/eesun/terser/data/audio/processed_Data/processed_data.csv'

        X,Y=[],[]
        for path,emotion,index in zip(main_df.File_Path,main_df.Emotion,range(main_df.File_Path.shape[0])):
            features=self.get_features(path)

            if index%500==0:
                print(f'{index} audio has been processed')

            for i in features:
                X.append(i)
                Y.append(emotion)
        print('Done')

        self.extract=pd.DataFrame(X)
        self.extract['Emotion']=Y
        self.extract.to_csv(self.processed_data_path,index=False)
        
        print("Preprocessed Data")
        print(self.extract.head(10))

        return self.extract
