import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display


class DataAnalyze:
    def __init__(self):

        self.Crema_Path='/home/eesun/terser/data/audio/Crema/'
        self.Ravdess_Path='/home/eesun/terser/data/audio/Ravdess/audio_speech_actors_01-24'
        self.Savee_Path='/home/eesun/terser/data/audio/Savee/'
        self.Tess_Path='/home/eesun/terser/data/audio/Tess/'

        self.crema = []
        self.ravdess = []
        self.savee = []
        self.tess = []

        self.audio_path = []
        self.colors={'disgust':'#804E2D',
                    'happy':'#F19C0E',
                    'sad':'#478FB8',
                    'neutral':'#4CB847',
                    'fear':'#7D55AA',
                    'angry':'#C00808',
                    'surprise':'#EE00FF'}
        

#%%
    def PreprocessingCrema(self):
        ## Crema
        """
            Crema 폴더 안에는 ex) 1001_DFA_ANG_XX.wav 와 같이
            감정 이름이 파일 이름으로 구성되어 있음.
            
            파일 이름에 해당되는 emotion label 부여하는 과정
        """
        for wav in os.listdir(self.Crema_Path):
            emotion=wav.partition(".wav")[0].split('_')
            if emotion[2]=='SAD':
                self.crema.append(('sad',self.Crema_Path+'/'+wav))
            elif emotion[2]=='ANG':
                self.crema.append(('angry',self.Crema_Path+'/'+wav))
            elif emotion[2]=='DIS':
                self.crema.append(('disgust',self.Crema_Path+'/'+wav))
            elif emotion[2]=='FEA':
                self.crema.append(('fear',self.Crema_Path+'/'+wav))
            elif emotion[2]=='HAP':
                self.crema.append(('happy',self.Crema_Path+'/'+wav))
            elif emotion[2]=='NEU':
                self.crema.append(('neutral',self.Crema_Path+'/'+wav))
            else:
                self.crema.append(('unknown',self.Crema_Path+'/'+wav))

        self.Crema_df=pd.DataFrame.from_dict(self.crema)
        self.Crema_df.rename(columns={0:'Emotion',1:'File_Path'},inplace=True)

        print('----Creama 전처리 후----')
        print(self.Crema_df.head())
        print('Crema data emotion label :', self.Crema_df['Emotion'].unique())

        return self.Crema_df
        
        
    def PreprocessingRavdess(self):
        ## Ravdess
        for directory in os.listdir(self.Ravdess_Path):
            actors=os.listdir(os.path.join(self.Ravdess_Path,directory))
            for wav in actors:
                emotion=wav.partition('.wav')[0].split('-')
                emotion_number=int(emotion[2])
                self.ravdess.append((emotion_number,os.path.join(self.Ravdess_Path,directory,wav)))
        
        self.Ravdess_df=pd.DataFrame.from_dict(self.ravdess)
        self.Ravdess_df.rename(columns={0:'Emotion',1:'File_Path'},inplace=True)
        self.Ravdess_df['Emotion'].replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'},inplace=True)
        
        print('----Ravdess 전처리 후----')
        print(self.Ravdess_df.head())
        print('Ravdess data emotion label :', self.Ravdess_df['Emotion'].unique())
        
        return self.Ravdess_df
    
        
    def PreprocessingSavee(self):
        ## savee
        for wav in os.listdir(self.Savee_Path):
            emo=wav.partition('.wav')[0].split('_')[1].replace(r'[0-9]','')
            emotion=re.split(r'[0-9]',emo)[0]
            if emotion=='a':
                self.savee.append(('angry',self.Savee_Path+'/'+wav))
            elif emotion=='d':
                self.savee.append(('disgust',self.Savee_Path+'/'+wav))
            elif emotion=='f':
                self.savee.append(('fear',self.Savee_Path+'/'+wav))
            elif emotion=='h':
                self.savee.append(('happy',self.Savee_Path+'/'+wav))
            elif emotion=='n':
                self.savee.append(('neutral',self.Savee_Path+'/'+wav))
            elif emotion=='sa':
                self.savee.append(('sad',self.Savee_Path+'/'+wav))
            elif emotion=='su':
                self.savee.append(('surprise',self.Savee_Path+'/'+wav))

        self.Savee_df=pd.DataFrame.from_dict(self.savee)
        self.Savee_df.rename(columns={0:'Emotion',1:'File_Path'},inplace=True)

        print('----Savee 전처리 후----')
        print(self.Savee_df.head())
        print('Savee data emotion label :', self.Savee_df['Emotion'].unique())
        
        return self.Savee_df

    def PreprocessingTess(self):
        ## tess
        for directory in os.listdir(self.Tess_Path):
            for wav in os.listdir(os.path.join(self.Tess_Path,directory)):
                emotion=wav.partition('.wav')[0].split('_')
                if emotion[2]=='ps':
                    self.tess.append(('surprise',os.path.join(self.Tess_Path,directory,wav)))
                else:
                    self.tess.append((emotion[2],os.path.join(self.Tess_Path,directory,wav)))
        
        self.Tess_df=pd.DataFrame.from_dict(self.tess)
        self.Tess_df.rename(columns={0:'Emotion',1:'File_Path'},inplace=True)

        print('----Tess 전처리 후----')
        print(self.Tess_df.head())
        print('Tess data emotion label :', self.Tess_df['Emotion'].unique())
        
        return self.Tess_df

    
    def GenerateTotalData(self):
        ## Crema + Ravdess + Savee + Tess
        self.main_df=pd.concat([self.Crema_df, self.Ravdess_df,self.Savee_df,self.Tess_df],axis=0)
        
        print('---------------------')
        print("Total data shape: ", self.main_df.shape)
        print(self.main_df.head(15))
        print(self.main_df.tail(15))

        return self.main_df

    def GetEmotionLabel(self):
        plt.figure(figsize=(12,6))
        plt.title('Emotions Counts')
        emotions=sns.countplot(x='Emotion',data=self.main_df,palette='Set2')
        emotions.set_xticklabels(emotions.get_xticklabels(),rotation=45)
        plt.show()

        self.emotion_names=self.main_df['Emotion'].unique()

        return self.emotion_names

    
    def GetAudioPath(self):
    
        def wave_plot(data,sr,emotion,color):
            plt.figure(figsize=(12,5))
            plt.title(f'{emotion} emotion for waveplot',size=17)
            librosa.display.waveshow(y=data,sr=sr,color=color)

        def spectogram(data,sr,emotion):
            audio=librosa.stft(data)
            audio_db=librosa.amplitude_to_db(abs(audio))
            plt.figure(figsize=(12,5))
            plt.title(f'{emotion} emotion for spectogram',size=17)
            librosa.display.specshow(audio_db,sr=sr,x_axis='time',y_axis='hz')

    
        for emotion in self.emotion_names:
            path=np.array(self.main_df['File_Path'][self.main_df['Emotion']==emotion])[1]
            data, sr=librosa.load(path)
            wave_plot(data, sr, emotion, self.colors[emotion])
            spectogram(data, sr, emotion)
            self.audio_path.append(path)

        return self.audio_path