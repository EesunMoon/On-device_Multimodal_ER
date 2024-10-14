import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.layers as L
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint 
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')


class SpeechEmotionRecognition:
    def __init__(self):
        '''
        '''
        ## default setting
        self.model_path = "./ser_model.h5"
        self.EPOCH=50
        self.BATCH_SIZE=64
    

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

        
    def ProcessedDataLoad(self, processed_data, flag):

        print('Processed Data Shape :', processed_data.shape)

        processed_data=processed_data.fillna(0)
        print('Is there any zero data?', processed_data.isna().any())
        print('Processed Data Shape :', processed_data.shape)
        print('Processed Data Head')
        print(processed_data.head(10))

        if flag == True:
            processed_data['Emotion'] = processed_data['Emotion'].apply(self.categoryTofour)
            print('Re-labeling Data Head')
            print(processed_data.head(10))

        return processed_data
    
    
    def GetTrainTestData(self,processed_data): 

        print('\nGet Train Test Data')

        x=processed_data.drop(labels='Emotion',axis=1)
        y=processed_data['Emotion']

        lb=LabelEncoder()
        y=np_utils.to_categorical(lb.fit_transform(y))
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

        return X_train, X_test, X_val, y_train, y_test, y_val
    

    def PreparationTrainTest(self, processed_data):
        # x, y data 불러오기
        x,y = self.GetTrainTestData(processed_data)

        # get Split data
        X_train, X_test, X_val, y_train, y_test, y_val = self.TrainTestSplit(x, y)

        # normalization process
        X_train, X_test, X_val, y_train, y_test, y_val = self.NormalizationData(X_train, X_test, X_val, 
                                                                                y_train, y_test, y_val)

        return X_train, X_test, X_val, y_train, y_test, y_val

    def ModelTrain(self,X_train, X_val, y_train, y_val, model_filename,
                   EPOCH, BATCH_SIZE, CATEGORY_NUM, model_dim, drop_rate):
        """
            model_dim : CNN의 dimension 뜻함
                model_dim = 0 :: 1D-CNN에 dropout 추가한 것
        """


        
        early_stop=EarlyStopping(monitor='val_accuracy',mode='auto',patience=5,restore_best_weights=True)
        lr_reduction=ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
        
        
        '''
        original version : 5D-CNN만 사용했을 때

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
                L.Dense(CATEGORY_NUM,activation='softmax')
            ])
        '''


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
                L.Dense(CATEGORY_NUM,activation='softmax')
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
                L.Dense(CATEGORY_NUM,activation='softmax')
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
                L.Dense(CATEGORY_NUM,activation='softmax')
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
                L.Dense(CATEGORY_NUM,activation='softmax')
            ])
        elif model_dim == 1:
            self.model=tf.keras.Sequential([
                L.Conv1D(256,kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
                L.BatchNormalization(),
                L.MaxPool1D(pool_size=5,strides=2,padding='same'),
                
                L.Flatten(),
                L.Dense(512,activation='relu'),
                L.BatchNormalization(),
                L.Dense(CATEGORY_NUM,activation='softmax')
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
                L.Dropout(drop_rate),   # add dropout 
                L.Dense(CATEGORY_NUM,activation='softmax')
            ])


        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='accuracy')
        self.model.summary()
        
        self.checkpointer = ModelCheckpoint(filepath = model_filename,
                                        monitor = 'val_accuracy',
                                        save_best_only = True, mode='max')
        '''
        self.history = self.model.fit(X_train, y_train, 
                                    epochs=EPOCH, validation_data=(X_val,y_val), 
                                    batch_size=BATCH_SIZE,
                                    verbose = 2, shuffle = True,
                                    callbacks = [self.checkpointer])
        '''
        self.history=self.model.fit(X_train, y_train, 
                        epochs=EPOCH, validation_data=(X_val,y_val), 
                        batch_size=BATCH_SIZE,callbacks=[early_stop,lr_reduction])
        
        self.PlottingHistory()

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


    def Prediction(self, X_test, y_test, emotion_names, confusion_prefix, model_filename):
        
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print('y pred :', y_pred)

        y_check=np.argmax(y_test,axis=1)
        print('y check :', y_check)

        loss,accuracy=self.model.evaluate(X_test, y_test,verbose=0)
        print(f'Test Loss: {loss}')
        print(f'Test Accuracy: {accuracy}')

        filename = confusion_prefix + str(format(accuracy, '.4f')) + '.csv'

        conf=confusion_matrix(y_check,y_pred)
        cm=pd.DataFrame(
            conf,index=[i for i in emotion_names],
            columns=[i for i in emotion_names]
        )
        pd.DataFrame(conf).to_csv(
            path_or_buf = filename, index = None, header = None)
        plt.figure(figsize=(12,7))
        ax=sns.heatmap(cm,annot=True,fmt='d')
        ax.set_title(f'confusion matrix for model ')
        plt.show()

        print(f'Model Confusion Matrix\n',classification_report(y_check,y_pred,
                                                            target_names=emotion_names))
        
        
        self.model.save(model_filename)
        

    def StartModelProcess(self, processed_data, emotion_names, flag, CATEGORY_NUM, model_dim,
                          EPOCH, BATCH_SIZE, drop_rate, model_dir, confusion_prefix, model_filename):

        # 1) processed data에 zero값 처리
        processed_data = self.ProcessedDataLoad(processed_data, flag)

        # 2) train, test, validation data 나누기 & normalization
        X_train, X_test, X_val, y_train, y_test, y_val = self.PreparationTrainTest(processed_data)

        # 3) model train
        self.ModelTrain(X_train, X_val, y_train, y_val, os.path.join(model_dir, model_filename),
                        EPOCH, BATCH_SIZE, CATEGORY_NUM, model_dim, drop_rate)

        if flag == True:
            emotion_names=processed_data['Emotion'].unique()

        
        # 4) prediction
        self.Prediction(X_test, y_test, emotion_names,
                        os.path.join(model_dir, confusion_prefix),
                        os.path.join(model_dir, model_filename))