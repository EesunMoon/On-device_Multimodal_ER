import sys, os 
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.metrics import log_loss
from keras.utils import plot_model

import warnings
warnings.filterwarnings('ignore')


class FusionWithSERHR:
    def __init__(self):
        # [default] name variables
        self.absolute_filepath          = '/home/mines/Desktop/Projects/Multimodal_Fusion/scorebased_fusion/'   # [수정]
        self.data_dir                   = 'data/'
        self.model_dir                  = 'model/'
        self.result_dir                 = 'result/'
        self.model_file_suffix          = '_model_shallow.hdf5'
        self.hr_format                  = 'HR'
        self.speech_format              = 'Speech'
        self.video_format               = 'Video'
        self.eeg_format                 = 'EEG'
        self.multimodal_format          = 'Multimodal'
        self.conf_prefix                = 'fusion_conf_'
        
        # data path variables
        self.format_csv                 = '.csv'
        self.format_npz                 = '.npz'
        self.hr_data_filedir            = self.absolute_filepath + self.data_dir + self.hr_format + '/'
        self.speech_data_filedir        = self.absolute_filepath + self.data_dir + self.speech_format + '/'
        self.video_data_filedir         = self.absolute_filepath + self.data_dir + self.video_format + '/'
        self.eeg_data_filedir           = self.absolute_filepath + self.data_dir + self.eeg_format + '/'
        self.multimodal_data_filedir    = self.absolute_filepath + self.data_dir + self.multimodal_format + '/'
        self.train_data_name            = 'train_dataset' + self.format_npz
        self.test_data_name             = 'test_dataset' + self.format_npz
        self.emotion_names              = ['happy', 'calm', 'sad', 'angry']

        # model path variables
        self.hr_model_filepath          = self.absolute_filepath + self.model_dir + self.hr_format + self.model_file_suffix
        self.speech_model_filepath      = self.absolute_filepath + self.model_dir + self.speech_format + self.model_file_suffix
        self.video_model_filepath       = self.absolute_filepath + self.model_dir + self.video_format + self.model_file_suffix
        self.eeg_model_filepath         = self.absolute_filepath + self.model_dir + self.eeg_format + self.model_file_suffix

        # model 가중치(performance)
        self.modality_num               = 4       # [수정] modality 개수
        self.hr_model_weight            = 0.42    # [수정]
        self.speech_model_weight        = 0.78    # [수정]
        self.video_model_weight         = 0.74    # [수정]
        self.eeg_model_weight           = 0.71    # [수정]

    #%%
    def LoadModel(self):

        print('\n------ LOAD Pre-Trained MODEL -----\n')
        
        self.hr_model = load_model(self.hr_model_filepath)
        print('HR_model LOAD COMPLETE')

        self.speech_model = load_model(self.speech_model_filepath)
        print('Speech_model LOAD COMPLETE')
        
        self.video_model = load_model(self.video_model_filepath)
        print('Video_model LOAD COMPLETE')

        self.eeg_model = load_model(self.eeg_model_filepath)
        print('EEG_model LOAD COMPLETE')

        print('\n------ ALL MODEL LOAD COMPLETE -----\n')

    def LoadData(self, isTrain):
        if isTrain == True:
            hr_data_filepath       = self.hr_data_filedir + self.train_data_name
            speech_data_filepath   = self.speech_data_filedir + self.train_data_name
            video_data_filepath    = self.video_data_filedir + self.train_data_name
            eeg_data_filepath      = self.eeg_data_filedir + self.train_data_name
        elif isTrain == False:
            hr_data_filepath       = self.hr_data_filedir + self.test_data_name
            speech_data_filepath   = self.speech_data_filedir + self.test_data_name
            video_data_filepath    = self.video_data_filedir + self.test_data_name
            eeg_data_filepath      = self.eeg_data_filedir + self.test_data_name

        
        hr_data                    = np.load(hr_data_filepath)
        speech_data                = np.load(speech_data_filepath)
        video_data                 = np.load(video_data_filepath)
        eeg_data                   = np.load(eeg_data_filepath)
        

        return hr_data, speech_data, video_data, eeg_data
    
    #%%
    def create_emotion_group(self, data):
        '''
            감정별 group 생성
        '''
        arg_y = np.argmax(data['y'], axis=1)

        happy_idx = np.where(arg_y == 0)[0]
        calm_idx = np.where(arg_y == 1)[0]
        sad_idx = np.where(arg_y == 2)[0]
        angry_idx = np.where(arg_y == 3)[0]

        happy = data['x'][list(happy_idx), :]
        calm = data['x'][list(calm_idx), :]
        sad = data['x'][list(sad_idx), :]
        angry = data['x'][list(angry_idx), :]

        return happy, calm, sad, angry

    def SplitDataForMinIdx(self, min_idx, hr_emo, speech_emo, video_emo, eeg_emo):
       
        hr = hr_emo[:min_idx, :]
        speech = speech_emo[:min_idx, :]
        video = video_emo[:min_idx, :]
        eeg = eeg_emo[:min_idx, :]

        return hr, speech, video, eeg
    
    def GroupingData(self, hr, speech, video, eeg):
       
        # 감정별 group 생성
        hr_happy, hr_calm, hr_sad, hr_angry = self.create_emotion_group(hr)
        speech_happy, speech_calm, speech_sad, speech_angry = self.create_emotion_group(speech)
        video_happy, video_calm, video_sad, video_angry = self.create_emotion_group(video)
        eeg_happy, eeg_calm, eeg_sad, eeg_angry = self.create_emotion_group(eeg)

        # 감정 min index 추출
        self.happy_min_idx = min(hr_happy.shape[0], speech_happy.shape[0], video_happy.shape[0], eeg_happy.shape[0])
        self.calm_min_idx = min(hr_calm.shape[0], speech_calm.shape[0], video_calm.shape[0], eeg_calm.shape[0])
        self.sad_min_idx = min(hr_sad.shape[0], speech_sad.shape[0], video_sad.shape[0], eeg_sad.shape[0])
        self.angry_min_idx = min(hr_angry.shape[0], speech_angry.shape[0], video_angry.shape[0], eeg_angry.shape[0])
        
        
        hr_happy, speech_happy, video_happy,eeg_happy = self.SplitDataForMinIdx(self.happy_min_idx, hr_happy, speech_happy, video_happy,eeg_happy)
        hr_calm, speech_calm, video_calm, eeg_calm = self.SplitDataForMinIdx(self.calm_min_idx, hr_calm, speech_calm, video_calm, eeg_calm)
        hr_sad, speech_sad, video_sad, eeg_sad = self.SplitDataForMinIdx(self.sad_min_idx, hr_sad, speech_sad, video_sad, eeg_sad)
        hr_angry, speech_angry, video_angry, eeg_angry = self.SplitDataForMinIdx(self.angry_min_idx, hr_angry, speech_angry, video_angry, eeg_angry)
        print("video_angry=", video_angry.shape, "eeg_angry=", eeg_angry.shape)

        hr_x = np.concatenate([hr_happy, hr_calm, hr_sad, hr_angry])
        speech_x = np.concatenate([speech_happy, speech_calm, speech_sad, speech_angry])
        video_x = np.concatenate([video_happy, video_calm, video_sad, video_angry])
        eeg_x = np.concatenate([eeg_happy, eeg_calm, eeg_sad, eeg_angry])

        label = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        y = [label[0]] * self.happy_min_idx + [label[1]] * self.calm_min_idx + [label[2]] * self.sad_min_idx + [label[3]] * self.angry_min_idx
        y = np.array(y)

        return hr_x, speech_x, video_x, eeg_x, y
    
    def SaveMultimodalData(self, hr_x, speech_x, video_x, eeg_x, y) :

        # save multi-modal data as npz format
        os.makedirs(self.multimodal_data_filedir, exist_ok = True)

        npz_file_name = self.multimodal_data_filedir + '/multimodal_dataset.npz'
        np.savez(npz_file_name, hr_x = hr_x, speech_x = speech_x, video_x = video_x, eeg_x = eeg_x, y = y)
        print('SAVED MULTI-MODAL DATASET.')


    #%%
    def ScoreBasedFusion(self, hr_pred, speech_pred, video_pred, eeg_pred):

        ## average & weight average   
        # 평균
        pred_avg = (hr_pred + speech_pred + video_pred + eeg_pred) / self.modality_num
        # 가중 평균
        pred_weight_avg = ((hr_pred * self.hr_model_weight) + (speech_pred * self.speech_model_weight) + (video_pred * self.video_model_weight) + (eeg_pred * self.eeg_model_weight)) / (self.hr_model_weight + self.speech_model_weight + self.video_model_weight + self.eeg_model_weight)
        
        pred_avg = np.argmax(pred_avg, axis=1)
        pred_weight_avg = np.argmax(pred_weight_avg, axis=1)
    
        return pred_avg, pred_weight_avg


    def Prediction(self, hr_x, speech_x, video_x, eeg_x, y):
        mode = ['average', 'weight_average']

        #-- model prediction --#
        print('Prediction !')
        hr_pred = self.hr_model.predict(hr_x, verbose=0)
        speech_pred = self.speech_model.predict(speech_x)
        video_pred = self.video_model.predict(video_x)
        eeg_pred = self.eeg_model.predict(eeg_x)

        #-- Fusion --#
        print('Score Based Fusion !')
        pred_avg, pred_weight_avg = self.ScoreBasedFusion(hr_pred, speech_pred, video_pred, eeg_pred)
        print(np.unique(pred_avg))
        print(np.unique(pred_weight_avg))
        y = np.argmax(y, axis=1)

        print('Get Performance !')
        print('AVGERAGE')
        self.GetPerformance(mode[0], y, pred_avg)
        print('Weighted-AVERAGE')
        self.GetPerformance(mode[1], y, pred_weight_avg)
        

    def GetPerformance(self, mode, y, pred):
        
        f1 = f1_score(y, pred, average='micro')
        print(mode, 'f1 score :', f1)
        

        filename_prefix = self.absolute_filepath + self.result_dir + self.conf_prefix + str(self.modality_num) + '_'
        filename = filename_prefix + mode + str(format(f1, '.4f')) + self.format_csv

        conf=confusion_matrix(y, pred)
        cm=pd.DataFrame(
            conf,index=[i for i in self.emotion_names],
            columns=[i for i in self.emotion_names]
        )
        pd.DataFrame(cm).to_csv(path_or_buf = filename)

        print(f'Model Confusion Matrix\n',classification_report(y, pred,
                                                            target_names=self.emotion_names))
        
        print()
        

    #%%
    def FusionProcess(self):
        # MODEL LOAD
        self.LoadModel()
        
        # DATA LOAD
        '''
            isTrain : True - train, False - test

            hr_train, speech_train, video_train, eeg_train = self.LoadData(isTrain)
            hr_test, speech_test, video_test, eeg_test = self.LoadData(isTrain)
        '''
        print('\n------ LOAD DATASET -----')
        print('\nTrain Dataset Load')
        isTrain = True
        hr_train, speech_train, video_train, eeg_train = self.LoadData(isTrain)
        print('hr train x :', hr_train['x'].shape)
        print('hr train y :', hr_train['y'].shape)
        print('speech train x :', speech_train['x'].shape)
        print('speech train y :', speech_train['y'].shape)
        print('video train x :', video_train['x'].shape)
        print('video train y :', video_train['y'].shape)
        print('eeg train x :', eeg_train['x'].shape)
        print('eeg train y :', eeg_train['y'].shape)

        print('\nTest Dataset Load')
        isTrain = False
        hr_test, speech_test, video_test, eeg_test = self.LoadData(isTrain)
        print('hr test x :', hr_test['x'].shape)
        print('hr test y :', hr_test['y'].shape)
        print('speech test x :', speech_test['x'].shape)
        print('speech test y :', speech_test['y'].shape)
        print('video test x :', video_test['x'].shape)
        print('video test y :', video_test['y'].shape)
        print('eeg test x :', eeg_test['x'].shape)
        print('eeg test y :', eeg_test['y'].shape)
        
        print('\n------ LOAD DATASET COMPLETE -----\n')


        # 아래 과정으로 min index 추출됨
        print('\n------ MAKE MULTIMODAL DATASET -----')
        # hr_x, speech_x, video_x, eeg_x, y = self.GroupingData(hr_train, speech_train, video_train, eeg_train)
        hr_x, speech_x, video_x, eeg_x, y = self.GroupingData(hr_test, speech_test, video_test, eeg_test)
        self.SaveMultimodalData(hr_x, speech_x, video_x, eeg_x, y)
        
        print('hr x :', hr_x.shape)
        print('speech x :', speech_x.shape)
        print('video x :', video_x.shape)
        print('eeg x :', eeg_x.shape)
        print('happy_min_idx :', self.happy_min_idx)
        print('calm_min_idx :', self.calm_min_idx)
        print('sad_min_idx :', self.sad_min_idx)
        print('angry_min_idx :', self.angry_min_idx)
        print('\n------ MAKE MULTIMODAL DATASET COMPLETE -----\n')

        print('\n------ SCORE-BASED FUSION -----')
        self.Prediction(hr_x, speech_x, video_x,eeg_x, y)
        print('\n------ SCORE-BASED FUSION COMPLETE -----\n')


if __name__ == "__main__":
    fusion = FusionWithSERHR()

    fusion.FusionProcess()
