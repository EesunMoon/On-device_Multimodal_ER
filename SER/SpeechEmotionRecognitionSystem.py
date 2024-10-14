from DataAnalyze import DataAnalyze
from DisplayEmotion import DisplayEmotion
from AudioAugmentation import AudioAugmentation
from FeatureExtraction import FeatureExtraction
from SpeechEmotionRecognition import SpeechEmotionRecognition

import pandas as pd

class SpeechEmotionRecognitionSystem:
    def __init__(self):
        '''
            Entire Process of Speech Emotion Recognition 
        '''

        self.model_dir = 'models_ser/'
        self.processed_data_path='/home/eesun/terser/data/audio/processed_Data/processed_data.csv'

        self.DataLoad()


#%%
    def DataLoad(self):
        """
            DataLoad() : 감정에 해당하는 path에 emotion label 부여
                        emotion name list 출력
                return
                    DataFrame()
                        0 : Emotion, 1 : File_Path

            
            #-- Data Info --#
                Crema : 7,442
                    emotion label : ['angry', 'neutral', 'disgust', 'sad', 'fear', 'happy'] #6
                Ravdess : 1,440
                    emotion label : ['angry', 'fear', 'disgust', 'sad', 'surprise', 'happy', 'neutral'] #7
                Savee : 480
                    emotion label : ['sad', 'neutral', 'surprise', 'fear', 'disgust', 'happy', 'angry'] #7
                Tess : 2,800
                    emotion label : ['disgust', 'surprise', 'happy', 'sad', 'neutral', 'fear', 'angry'] #7
                
                Total (Crema + Ravdess + Savee + Tess) : (12162, 2)
        """

        # 파일에 해당하는 감정 labeling
        da = DataAnalyze()
        self.Crema_df = da.PreprocessingCrema()
        self.Ravdess_df = da.PreprocessingRavdess()
        self.Savee_df = da.PreprocessingSavee()
        self.Tess_df = da.PreprocessingTess()
        self.main_df = da.GenerateTotalData()

        self.emotion_names = da.GetEmotionLabel()
        self.audio_path = da.GetAudioPath()

        # 전처리 한 결과 display
        de = DisplayEmotion()
        de.DisplayAudioEmotion(self.audio_path)


        # 처음부터 시작
        # self.AudioProcess()

        # processed 이후부터 시작
        self.processed_data = pd.read_csv(self.processed_data_path)
        self.SpeechEmotionRecognitionProcess()

    

#%%
    def AudioProcess(self):
        """
            AudioProcess() : 오디오 파일에 잡음 추가 - Audio Augmentation
                noise 추가
                stretching
                shift (시간 이동)
                pitching
        """

        aa = AudioAugmentation()
        self.data, self.sr = aa.LoadAudioData(self.audio_path)
        
        # audio 원본에 noise 추가
        self.noised_audio, self.stretched_audio, self.shifted_audio, self.pitched_audio = aa.GenerateAudioAugmentation(self.data, self.sr)

        '''
        de = DisplayEmotion()
        de.DisplayAudioAugmentation(self.data, self.audio_path, self.sr, 
                                    self.noised_audio, self.stretched_audio, self.shifted_audio, self.pitched_audio)
        '''
        self.FeatureExtractionProcess()


#%%    
    def FeatureExtractionProcess(self):
        """
            FeatureExtractionProcess() : Audio Vector 화
                1개의 extract_feature
                    zcr
                    rms
                    mfcc
                    ---------
                    parameter
                        frame_length = 2,048 (계산을 수행할 프레임의 길이)
                        hop_length = 512 (각 프레임에 대해 진행할 sample 수)
            
                
            total featured audio data
                1. original audio
                2. noised audio
                3. pitched audio
                4. pitched-noised audio
            
                (12162, 2) x 4 = 48,648
                
                    offset = 0.6
                    duration = 2.5

                    즉, 하나의 오디오 파일 당 2.5씩의 time 구간을 가짐

        """

        fe = FeatureExtraction()
        self.processed_data = fe.SaveProcessedData(self.main_df)

        self.SpeechEmotionRecognitionProcess()

#%%
    def SpeechEmotionRecognitionProcess(self):
        """
            train & test
        """
        confusion_prefix = 'result_conf_ser'
        # model_filename = "ser_model_category4.h5"
        model_filename = "ser_model_"
        '''
            category change
            flag:: 4개 category로 할거면 flag = True, 그대로면 flag = False
            CATEGORY_NUM : category개수
        '''

        flag = True             # default = False(category가 7개 그대로 일 때)
        CATEGORY_NUM = 7        # default = 7
        EPOCH=50
        BATCH_SIZE=64
        drop_rate = 0.5

        if flag == True:
            CATEGORY_NUM = 4


        '''
            model dimension change
            default : 5D-CNN
                5 : 5D-CNN
                4 : 4D-CNN
                3 : 3D-CNN
                2 : 2D-CNN
                1 ; 1D-CNN
                0 : 1D-CNN + dropout
        '''
        model_dim = 0 # model dimension

        model_filename = model_filename + str(model_dim) + 'D_try.hdf5'
        ser = SpeechEmotionRecognition()
        ser.StartModelProcess(self.processed_data, self.emotion_names, flag, CATEGORY_NUM, model_dim,
                              EPOCH, BATCH_SIZE, drop_rate, self.model_dir, confusion_prefix, model_filename)
        
#%%         
if __name__ == "__main__":
    ser = SpeechEmotionRecognitionSystem()