import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as L
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint 
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler
import collections
# import matplotlib.pyplot as plt
# import plotly.express as px
# import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')


class SpeechEmotionRecognition:
    def __init__(self):
        '''
        '''
        ## default setting
        self.emotion_names = ['happy','calm','sad','angry']
        self.CATEGORY_NUM = 4

        self.absolute_data_filepath = '/Users/eesun/CODE/LAB/fusion/data/speech/'
        '''
        self.x_train_filepath       = self.absolute_data_filepath + 'train_dataset/' + 'speech_train.npy'
        self.y_train_filepath       = self.absolute_data_filepath + 'train_dataset/' + 'emotion_train.npy'
        self.x_val_filepath         = self.absolute_data_filepath + 'validation_dataset/' + 'speech_val.npy'
        self.y_val_filepath         = self.absolute_data_filepath + 'validation_dataset/' + 'emotion_val.npy'
        '''
        self.speech_data_filepath   = self.absolute_data_filepath + 'processed_data.csv'                    # processed speech data filepath
        self.absolute_filepath      = '/Users/eesun/CODE/LAB/fusion/speech/'                                # result(model & prediction) save filepath 
        self.model_dir              = self.absolute_filepath + 'models_ser/'
        self.saved_model_dir        = self.absolute_filepath + 'models_ser_prior/' + 'ser_model_category4.h5'                                    
        self.instance_flag          = '_instance'
        self.confusion_prefix       = 'result_conf_ser' + self.instance_flag
        self.EPOCH                  = 50
        self.BATCH_SIZE             = 64
        self.drop_rate              = 0.5
        self.weight                 = [0.3, 0.1, 0.3, 0.3]
        self.seq_num                = 6


    def categoryTofour(self, emotion):
        '''
            기존의 angry, disgust, fear, happy, neutral, sad, surprise의 7개 category를
                angry + disgust + fear => angry
                happy + surprise => happy
                neutral => calm
                sad => sad
            로 4개 category로 변환
        '''
        if emotion == 'fear' or emotion =='disgust':
            return 'angry'
        elif emotion == 'surprise' :
            return 'happy'
        elif emotion == 'neutral':
            return 'calm'
        else:
            return emotion

        
    def DataLoad(self):
        '''
            DataLoad : feature extraction 된 speech 데이터 Load
                한 개의 열 당 2.5sec의 instance
        '''
        speech_data = pd.read_csv(self.speech_data_filepath)
        print('Data Shape :', speech_data.shape)
        print('Data Description :', speech_data.describe())

        # fill Nan data for zero value
        speech_data=speech_data.fillna(0)
        print('Is there any zero data?', speech_data.isna().any())
        print('Processed Data Shape :', speech_data.shape)
        print('Processed Data Description :', speech_data.describe())
        print('Processed Data Head')
        print(speech_data.head(10))

        # Re-labeling Data : 7 categories >> 4 categories(Happy, Calm, Sad, Angry)
        speech_data['Emotion'] = speech_data['Emotion'].apply(self.categoryTofour)
        print('Count Emotions :', collections.Counter(speech_data['Emotion']))
        print('Re-labeling Data Head')
        print(speech_data.head(10))

        return speech_data
    
    
    def GetTrainTestData(self,speech_data): 
        '''
            GetTrainTestData : Split speech Data for X, Y
        '''

        print('\nGet Train Test Data')

        x = speech_data.drop(labels='Emotion',axis=1)
        y = speech_data['Emotion']

        lb = LabelEncoder()
        y = np_utils.to_categorical(lb.fit_transform(y))
        print('y classes :', lb.classes_)
        print('y label :', y)
        
        return x, y

    
    def TrainTestSplit(self, x, y):

        X_train,X_test,y_train,y_test=train_test_split(x, y, random_state=42,test_size=0.2,shuffle=True)

        print('** train & test data shape')
        print('x train :', X_train.shape)
        print('x test :', X_test.shape)
        print('y trian :', y_train.shape)
        print('y test :', y_test.shape)
        

        X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,random_state=42,test_size=0.1,shuffle=True)
        print('** train & test & validation data shape')
        print('x train :', X_train.shape)
        print('x test :', X_test.shape)
        print('x val :', X_val.shape)
        print('y train :', y_train.shape)
        print('y test :', y_test.shape)
        print('y val :', y_val.shape)
    
        return X_train, X_test, X_val, y_train, y_test, y_val


    def NormalizationData(self, X_train, X_test, X_val, y_train, y_test, y_val):
        print("\n----- Normalization -----")

        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train)
        X_test=scaler.transform(X_test)
        X_val=scaler.transform(X_val)

        print('** Normalized train & test & validation data shape')
        print('x train :', X_train.shape)
        print('x test :', X_test.shape)
        print('x val :', X_val.shape)
        print('y train :', y_train.shape)
        print('y test :', y_test.shape)
        print('y val :', y_val.shape)

        X_train=np.expand_dims(X_train,axis=2)
        X_val=np.expand_dims(X_val,axis=2)
        X_test=np.expand_dims(X_test,axis=2)
        print('** expanded x data shape')
        print('x train :', X_train.shape)
        print('x test :', X_test.shape)
        print('x val :', X_val.shape)

        print('----- Complete Normalization -----\n')

        return X_train, X_test, X_val, y_train, y_test, y_val
    
    def PreparationTrainTest(self, speech_data):

        # split speech for X, Y data
        x, y = self.GetTrainTestData(speech_data)

        # split Train, Test, Validation Data
        '''
            Validation dataset : model 학습 시 check point
        '''
        X_train, X_test, X_val, y_train, y_test, y_val = self.TrainTestSplit(x, y)

        # Normalization process
        X_train, X_test, X_val, y_train, y_test, y_val = self.NormalizationData(X_train, X_test, X_val, 
                                                                                y_train, y_test, y_val)

        return X_train, X_test, X_val, y_train, y_test, y_val
    

    def ModelTrain(self,X_train, X_val, y_train, y_val, model_dim, model_filename):
        """
            model_dim : CNN의 dimension 뜻함
                model_dim = 0 :: 1D-CNN에 dropout 추가한 것
        """

        early_stop=EarlyStopping(monitor='val_accuracy',mode='auto',patience=5,restore_best_weights=True)
        lr_reduction=ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.00001)


        if model_dim == 5:
            self.model=tf.keras.Sequential([
                L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                L.Conv1D(512,kernel_size=5,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                L.Conv1D(256,kernel_size=5,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=3,strides=2,padding='same'),
                L.Flatten(),
                L.Dense(512,activation='relu'),
                L.BatchNormalization(),
                L.Dense(self.CATEGORY_NUM,activation='softmax')
            ])
        elif model_dim == 4:
            self.model=tf.keras.Sequential([
                L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                
                L.Conv1D(512,kernel_size=5,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),

                L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),

                L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=3,strides=2,padding='same'),

                L.Flatten(),
                L.Dense(512,activation='relu'),
                L.BatchNormalization(),
                L.Dense(self.CATEGORY_NUM,activation='softmax')
            ])
        elif model_dim == 3:
            self.model=tf.keras.Sequential([
                L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                
                L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),

                L.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=3,strides=2,padding='same'),

                L.Flatten(),
                L.Dense(512,activation='relu'),
                L.BatchNormalization(),
                L.Dense(self.CATEGORY_NUM,activation='softmax')
            ])
        elif model_dim == 2:
            self.model=tf.keras.Sequential([
                L.Conv1D(512,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                
                L.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),

                L.Flatten(),
                L.Dense(512,activation='relu'),
                L.BatchNormalization(),
                L.Dense(self.CATEGORY_NUM,activation='softmax')
            ])
        elif model_dim == 1:
            self.model=tf.keras.Sequential([
                L.Conv1D(256,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                
                L.Flatten(),
                L.Dense(512,activation='relu'),
                L.BatchNormalization(),
                L.Dense(self.CATEGORY_NUM,activation='softmax')
            ])
        elif model_dim == 0:
            self.model=tf.keras.Sequential([
                L.Conv1D(128,kernel_size=10, strides=1,padding='same', activation='relu',
                         input_shape=(X_train.shape[1],1)),
                # L.BatchNormalization(),
                L.MaxPool1D(pool_size=2),
                
                L.Flatten(),
                L.Dense(256,activation='relu'),
                # L.BatchNormalization(),
                L.Dropout(self.drop_rate),   # add dropout 
                L.Dense(self.CATEGORY_NUM,activation='softmax')
            ])


        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
        self.model.summary()
        
        self.checkpointer = ModelCheckpoint(filepath = model_filename,
                                        monitor = 'val_accuracy',
                                        save_best_only = True, mode='max')
        
        self.history=self.model.fit(X_train, y_train, 
                        epochs=self.EPOCH, validation_data=(X_val,y_val), 
                        batch_size=self.BATCH_SIZE,callbacks=[early_stop,lr_reduction])
        
        
        # self.PlottingHistory()

        return self.model
        

    def PlottingHistory(self):
        fig=px.line(self.history.history,y=['accuracy','val_accuracy'],
           labels={'index':'epoch','value':'accuracy'},
           title=f'According to the epoch accuracy and validation accuracy chart for the model')
        fig.show()

        fig=px.line(self.history.history,y=['loss','val_loss'],
           labels={'index':'epoch','value':'loss'},
           title=f'According to the epoch loss and validation loss chart for the model')
        fig.show()

    def LoadModel(self):

        # 감정 예측 모델
        print('\n------ LOAD SAVED MODEL -----')
        self.model = load_model(self.saved_model_dir)

    def Softmax(self, x):
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x)
        y = exp_x / sum_exp_x

        return y

    def create_sequece(self, y_pred):
        pred_avg = []
        pred_weight_avg = []
        stop_idx = int(y_pred / self.seq_num) * self.seq_num - 1

        for i in range(0, y_pred.shape[0] - self.seq_num, self.seq_num):

            if i == stop_idx:
                break

            # seq_num개씩 prediction 짜르기
            seq = y_pred[i:i+self.seq_num] 
            
            # 평균
            avg = np.average(seq, axis=1) 
            pred_avg.append(avg)

            # 가중 평균
            weight_avg = np.average(seq, axis=1, weights = self.weight)
            pred_weight_avg.append(weight_avg)
        
        return pred_avg, pred_weight_avg
    
    def ScoreBasedFusion(self, y_pred):
        '''
            1) 단순히 argmax 결과로 voting
            2) 6개 instance의 softmax 평균 구한 후 최종 argmax
            3) 6개 instance의 softmax 가중평균 구한 후 최종 argmax
        '''

        y_pred1 = np.argmax(y_pred, axis=1)
        '''
        y_pred2 = np.average(y_pred, axis=1)
        y_pred3 = np.average(y_pred, axis=1, weights = weight)
        '''
        y_pred2, y_pred3 = self.create_sequece(y_pred)
        
        y_pred2 = np.argmax(y_pred2, axis=1)
        y_pred3 = np.argmax(y_pred3, axis=1)

        return y_pred1, y_pred2, y_pred3
    
    def SaveConfusionMatrix(self, confusion_prefix, mode, y_check, y_pred):
        filename = confusion_prefix + mode + str(format(accuracy, '.4f')) + '.csv'

        conf=confusion_matrix(y_check,y_pred)
        cm=pd.DataFrame(
            conf,index=[i for i in self.emotion_names],
            columns=[i for i in self.emotion_names]
        )
        pd.DataFrame(conf).to_csv(
            path_or_buf = filename, index = None, header = None)
        plt.figure(figsize=(12,7))
        ax=sns.heatmap(cm,annot=True,fmt='d')
        ax.set_title(f'confusion matrix for model ')
        plt.show()

        print('\n-----', mode, '-----')
        print(f'Model Confusion Matrix\n',classification_report(y_check,y_pred,
                                                            target_names=self.emotion_names))


    def Prediction(self, X_test, y_test, confusion_prefix, model_filename):
        mode = ['argmax', 'average', 'weight_average']

        y_pred = self.model.predict(X_test)
        # y_pred = np.argmax(y_pred, axis=1)
        # y_pred = self.Softmax(y_pred)
        y_pred1, y_pred2, y_pred3 = self.ScoreBasedFusion(y_pred)
        print('y pred based argmax:', y_pred1)
        print('y pred based average :', y_pred2)
        print('y pred based weight average :', y_pred3)

        # y_check=np.argmax(y_test,axis=1)
        # y_check = self.Softmax(y_check)
        y_check1, y_check2, y_check3 = self.ScoreBasedFusion(y_test)
        print('y check based argmax:', y_check1)
        print('y check based average :', y_check2)
        print('y check based weight average :', y_check3)

        loss,accuracy=self.model.evaluate(X_test, y_test,verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')

        
        self.SaveConfusionMatrix(confusion_prefix, mode[0], y_check1, y_pred1)
        self.SaveConfusionMatrix(confusion_prefix, mode[1], y_check2, y_pred2)
        self.SaveConfusionMatrix(confusion_prefix, mode[2], y_check3, y_pred3)

        '''
        filename = confusion_prefix + str(format(accuracy, '.4f')) + '.csv'

        conf=confusion_matrix(y_check,y_pred)
        cm=pd.DataFrame(
            conf,index=[i for i in self.emotion_names],
            columns=[i for i in self.emotion_names]
        )
        pd.DataFrame(conf).to_csv(
            path_or_buf = filename, index = None, header = None)
        plt.figure(figsize=(12,7))
        ax=sns.heatmap(cm,annot=True,fmt='d')
        ax.set_title(f'confusion matrix for model ')
        plt.show()

        print(f'Model Confusion Matrix\n',classification_report(y_check,y_pred,
                                                            target_names=self.emotion_names))
        '''
        
        self.model.save(model_filename)
        

    def StartModelProcess(self, model_dim, model_filename):
        
        # 1) Processed Speech Data Load & fill Nan value for zero
        speech_data = self.DataLoad()

        # 2) Split << Train, Test, Validation >> data & Normalization
        X_train, X_test, X_val, y_train, y_test, y_val = self.PreparationTrainTest(speech_data)

        '''
        # 3-1) model train
        self.ModelTrain(X_train, X_val, y_train, y_val, 
                        model_dim, os.path.join(self.model_dir, model_filename))
        '''

        # 3-2) trained model load
        self.LoadModel()

        # 4) prediction
        self.Prediction(X_test, y_test,
                        os.path.join(self.model_dir, self.confusion_prefix),
                        os.path.join(self.model_dir, model_filename))

#%%         
if __name__ == "__main__":
    ser = SpeechEmotionRecognition()
    
    model_dim = 2
    model_filename = "ser_model_" + 'instance_'
    model_filename = model_filename + str(model_dim) + 'D.hdf5'
    ser.StartModelProcess(model_dim, model_filename)
    