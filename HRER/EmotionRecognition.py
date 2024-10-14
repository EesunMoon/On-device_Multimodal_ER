from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from keras.callbacks import ModelCheckpoint 
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.layers import Dense, LSTM, Flatten
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import TomekLinks 
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from keras.utils import to_categorical
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import fwr13y.d9m.tensorflow as tf_determinism
import random
import seaborn as sns
import matplotlib.pyplot as plt
from MakeOurSamples import MakeOurSamples
from MakeSensorsSamples import MakeSensorsSamples
from MakeUlmTsstSamples import MakeUlmTsstSamples

    

class EmotionRecognition :


    def __init__(self) :

        self.project_dir                  = '/home/mines/Desktop/Projects/EmotionRecognition_HeartRate/GPU/'
        self.model_dir                    = self.project_dir + 'models/'                      
        self.number_of_classes            = 4
        self.filters                      = 128                            
        self.kernel_sizes                 = [10, 6, 1, 1]                
        self.pool_sizes                   = [2, 1, 1]                      
        self.drop_rate                    = 0.5                           
        self.neurons                      = [256, 128, 64]         
        self.batch                        = 32                     
        self.number_epochs                = 20                     
        self.model_filename               = 'model_-15s_0s.hdf5'  
        self.history_filename             = 'history_-15s_0s.csv'  
        self.pred_prefix                  = 'preds_-15s_0s'        
        self.confusion_prefix             = 'result_conf_-15s_0s'
        self.modelType                    = 0                               # select : 0 (CNN_Flatten) or (CNN LSTM) or 2(DCNN)
        self.cluster_balance              = 0.05                   

        self.ours_emotion_file_path       = self.project_dir + 'data/EMOTION_khy+taein+mes_new.csv'
        self.ours_hr_file_path            = self.project_dir + 'data/HRDATA_khy+taein+mes_new.csv'

        self.sensors_emotion_file_path    = self.project_dir + 'data/EMOTION_sensors_new.csv'
        self.sensors_hr_file_path         = self.project_dir + 'data/HRDATA_sensors_selected_5sec_and_interpolated_1sec.csv'

        self.ult_data_file_path           = self.project_dir + 'data/c3_muse_stress_2022_total_new.csv'

        self.back_second                  = 15
        self.after_second                 = 0

        self.removing_outliers_base       = 'variance'                      # select : 'mean' or 'variance' or 'both'
        self.histogram_base               = 'mean'                          # select : 'mean' or 'variance'
        self.oversampling_method          = 'kmeans'                        # select : 'kmeans' or 'random'

    
    # model
    def create_model(self, X_train, X_val, Y_train, Y_val, number_of_classes = 5, filters = 128, kernel_size = [10, 4, 1, 1],
                    pool_size = [2, 1, 1], drop_rate = 0.5, neurons = [256, 128, 64],
                    model_filename = 'best_model.hdf5', history_filename = 'history.csv',
                    batch = 32, number_epochs = 100, modelType = 0):

        if modelType == 0:            # Test model 1: 1D CNN Fatten
            model = Sequential()
            model.add(Conv1D(filters, kernel_size[1], activation = 'relu', input_shape = X_train.shape[1:]))  
            model.add(Dropout(drop_rate))             
            model.add(MaxPooling1D(pool_size[0]))
            model.add(Flatten())
            model.add(Dense(neurons[0], kernel_initializer = 'normal', activation = 'relu'))            
            model.add(Dropout(drop_rate))         

        if modelType == 1:            # Test model 2: 1D CNN LSTM         
            
            model = Sequential()
            model.add(Conv1D(filters, kernel_size[1], activation='relu', input_shape = X_train.shape[1:]))
            model.add(MaxPooling1D(pool_size[0]))    
            model.add(Dropout(drop_rate))
            model.add(LSTM(neurons[1] * 2, activation = 'relu'))    
            model.add(Dropout(drop_rate))                         
            
        if modelType == 2:            # Test model 3: DCNN
            model = Sequential()
            model.add(Conv1D(filters, kernel_size[1], activation = 'relu', input_shape = X_train.shape[1:]))
            model.add(MaxPooling1D(pool_size[0]))
            model.add(Dropout(drop_rate))   

            model.add(Conv1D(filters, kernel_size[2], activation = 'relu'))
            model.add(MaxPooling1D(pool_size[1]))
            model.add(Dropout(drop_rate))

            model.add(Conv1D(filters, kernel_size[3], activation = 'relu'))
            model.add(MaxPooling1D(pool_size[2]))
            model.add(Dropout(drop_rate))

            model.add(Conv1D(filters, kernel_size[3], activation = 'relu'))
            model.add(GlobalAveragePooling1D())
            
            model.add(Dense(neurons[0], kernel_initializer = 'normal', activation = 'relu'))
            model.add(Dropout(drop_rate))

            model.add(Dense(neurons[1], kernel_initializer = 'normal', activation = 'relu'))
            model.add(Dropout(drop_rate))

            model.add(Dense(neurons[2], kernel_initializer = 'normal', activation = 'relu'))
            model.add(Dropout(drop_rate))
        
        model.add(Dense(number_of_classes, kernel_initializer = 'normal', activation = 'softmax'))
        optimize = Adam(learning_rate = 0.001)
        model.compile(loss = 'categorical_crossentropy', optimizer = optimize, metrics = ['accuracy'])
        model.summary()
        
        checkpointer = ModelCheckpoint(filepath = model_filename, monitor = 'val_accuracy', verbose = 1, save_best_only = False, mode = 'max')
        
        hist = model.fit(X_train, Y_train,
                        validation_data = (X_val, Y_val),
                        batch_size = batch, epochs = number_epochs,
                        verbose = 2, shuffle = True,
                        callbacks = [checkpointer])
        
        pd.DataFrame(hist.history).to_csv(path_or_buf = history_filename)

        return model


    # convert softmax value to predicted label number
    def convert_vector(self, x):

        label_mapped = np.zeros((np.shape(x)[0]))
        for i in range(np.shape(x)[0]):
            max_value = max(x[i, : ])
            max_index = list(x[i, : ]).index(max_value)
            label_mapped[i] = max_index
            
        return label_mapped.astype(int)


    # prediction
    def prediction(self, X_val, Y_val, model, pred_prefix = 'preds_',  confusion_prefix = 'result_conf'):

        label_num = [0, 1, 2, 3]

        # val 데이터 성능 평가. 
        predictions = model.predict(X_val)
        df_predictions = pd.DataFrame(predictions)
        df_predictions['1st'] = np.max(predictions, axis = 1)
        df_predictions.to_csv(self.model_dir + 'predictions.csv', index = False)
        
        score = accuracy_score(self.convert_vector(Y_val), self.convert_vector(predictions))
        print('Last epoch\'s validation score is ', score)
        
        f1 = f1_score(self.convert_vector(Y_val), self.convert_vector(predictions), pos_label = 1, average = 'micro')        
        print('The f1-score of classifier on validation data is ', f1)
                    
        df = pd.DataFrame(self.convert_vector(predictions))
        filename = pred_prefix + '_val_' + str(format(score, '.4f')) + '.csv'
        df.to_csv(path_or_buf = filename, index = None, header = None)

        filename = confusion_prefix + '_val_' + str(format(score, '.4f')) 
        filename += '.csv'
        matrix = confusion_matrix(self.convert_vector(Y_val), self.convert_vector(predictions), labels = label_num)
        pd.DataFrame(matrix).to_csv(path_or_buf = filename, index = None, header = None)


        # class 별로 f1-score 확인
        print('------validation confusion matrix------')
        print(matrix)
        print(classification_report(self.convert_vector(Y_val), self.convert_vector(predictions), labels = label_num))
        print()

        return score


    # Oversampling + Cleaning
    def oversampling(self, X_array, Y_array, method, strategy) :

        if method == 'kmeans' :
            # kmeans oversampling + cleaning
            sm = KMeansSMOTE(sampling_strategy = strategy, random_state = 0, cluster_balance_threshold = self.cluster_balance) 
            X_res_clean, Y_res_clean = sm.fit_resample(X_array, Y_array)
            sm = TomekLinks()    
            X_res, Y_res = sm.fit_resample(X_res_clean, Y_res_clean)

        if method == 'knn' :
            # knn oversampling
            sm = SMOTE(sampling_strategy = strategy, random_state = 0, k_neighbors = 5)
            X_res, Y_res = sm.fit_resample(X_array, Y_array)

        if method == 'random' :
            # random oversampling
            sm = RandomOverSampler(sampling_strategy = strategy, random_state = 42)
            X_res, Y_res = sm.fit_resample(X_array, Y_array)

        print('X_res shape :', X_res.shape)
        print('Y_res shape :', Y_res.shape)

        return (X_res, Y_res)
    

    # Undersampling
    def undersampling(self, X_array, Y_array, strategy) :

        # random undersampling
        sm = RandomUnderSampler(sampling_strategy = strategy, random_state = 42)
        X_res, Y_res = sm.fit_resample(X_array, Y_array)

        print('X_res shape :', X_res.shape)
        print('Y_res shape :', Y_res.shape)

        return (X_res, Y_res)


    # Create DataFrame
    def create_dataframe(self, X_res, Y_res) :

        sample_list = []
        for X, Y in zip(X_res.tolist(), Y_res.tolist()) :
            sample = X + [Y]
            sample_list.append(sample)

        column_name = []
        for i in range(0, self.back_second + self.after_second) :
            name = 'hr' + str(i)
            column_name.append(name)
        column_name.append('emotion')

        df_new = pd.DataFrame(sample_list, columns = column_name)
        df_new['mean_hr'] = df_new.iloc[:, :-1].mean(axis = 1).to_numpy()
        print(df_new)

        return df_new


    # Plot histograms
    def plot_histogram(self, participant_name, df_new) :

        # plot histogram for each emotion
        print(f'-----Plot histograms for {participant_name} data-----')

        # emotion_list = df_new['emotion'].unique().tolist()
        emotion_dic = {0 : 'happy', 1 : 'calm', 2 : 'sad', 3 : 'angry'}
        emotion_list = [0, 1, 2, 3]

        if self.histogram_base == 'mean' :
            for emotion in emotion_list :
                df_emotion = df_new.loc[df_new['emotion'] == emotion]
                total_frequency = len(df_emotion)
                print(f"Total Frequency of '{emotion_dic[emotion]}' Emotion : {total_frequency}")

                plt.figure()
                plt.title(f"Mean of Heart Rate Distribution for '{emotion_dic[emotion]}' Emotion")
                plt.xlabel('Mean of Heart Rate')
                plt.ylabel('Frequency')
                sns.histplot(data = df_emotion, x = 'mean_hr', kde = True)
                # plt.show()
                file_dir = self.project_dir + 'data analysis result/EmotionRecognition.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/oversampling method - ' + self.oversampling_method + '/historgram base - ' + self.histogram_base + '/'
                os.makedirs(file_dir, exist_ok = True)
                plt.savefig(file_dir + emotion_dic[emotion] + '.png')
                plt.close()

            # plot histogram for all emotions
            plt.figure()
            plt.title(f"Mean of Heart Rate Distribution for All Emotions")
            plt.xlabel('Mean of Heart Rate')
            plt.ylabel('Frequency')
            plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            sns.histplot(data = df_new, x = 'mean_hr', kde = True, hue = 'emotion', palette = 'bright')
            # plt.show()
            file_dir = self.project_dir + 'data analysis result/EmotionRecognition.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/oversampling method - ' + self.oversampling_method + '/historgram base - ' + self.histogram_base
            os.makedirs(file_dir, exist_ok = True)
            plt.savefig(file_dir + '/all emotions.png')
            plt.close()

        if self.histogram_base == 'variance' :
            for emotion in emotion_list :
                df_emotion = df_new.loc[df_new['emotion'] == emotion]
                total_frequency = len(df_emotion)
                print(f"Total Frequency of '{emotion_dic[emotion]}' Emotion : {total_frequency}")

                plt.figure()
                plt.title(f"Variance of Heart Rate Distribution for '{emotion_dic[emotion]}' Emotion")
                plt.xlabel('Variance of Heart Rate')
                plt.ylabel('Frequency')
                sns.histplot(data = df_emotion, x = 'variance_hr', kde = True)
                # plt.show()
                file_dir = self.project_dir + 'data analysis result/EmotionRecognition.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/oversampling method - ' + self.oversampling_method + '/historgram base - ' + self.histogram_base + '/'
                os.makedirs(file_dir, exist_ok = True)
                plt.savefig(file_dir + emotion_dic[emotion] + '.png')

                plt.close()

            plt.figure()
            plt.title(f"Variance of Heart Rate Distribution for All Emotions")
            plt.xlabel('Variance of Heart Rate')
            plt.ylabel('Frequency')
            plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            sns.histplot(data = df_new, x = 'variance_hr', kde = True, hue = 'emotion', palette = 'bright')
            # plt.show()
            file_dir = self.project_dir + 'data analysis result/EmotionRecognition.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/oversampling method - ' + self.oversampling_method + '/historgram base - ' + self.histogram_base
            os.makedirs(file_dir, exist_ok = True)
            plt.savefig(file_dir + '/all emotions.png')
            plt.close()

        print('-----------------------------------------\n')


    # Create train data
    def create_train_data(self, df_ours_new, df_sensors_new, df_ulm_tsst_new) :

        # create train data
        # ours
        df_ours_new = df_ours_new.replace({'happy' : 0, 'calm' : 1, 'sad' : 2, 'angry' : 3})
        X_array_ours = df_ours_new.iloc[:, 1:-3].to_numpy()
        Y_array_ours = df_ours_new.iloc[:, -3].to_numpy()

        # sensors
        df_sensors_new = df_sensors_new.replace({'happy' : 0, 'calm' : 1, 'sad' : 2, 'angry' : 3})
        X_array_sensors = df_sensors_new.iloc[:, 1:-3].to_numpy()
        Y_array_sensors = df_sensors_new.iloc[:, -3].to_numpy()

        # ulm-tsst
        df_ulm_tsst_new = df_ulm_tsst_new.replace({'happy' : 0, 'calm' : 1, 'sad' : 2, 'angry' : 3})
        X_array_ulm_tsst = df_ulm_tsst_new.iloc[:, 1:-3].to_numpy()
        Y_array_ulm_tsst = df_ulm_tsst_new.iloc[:, -3].to_numpy()


        # oversampling -> merge
        # ours
        # undersample the majority class to 700
        calm_idx = np.where(Y_array_ours == 1)[0]
        other_idx = np.where(Y_array_ours != 1)[0]
        calm_idx = np.random.choice(calm_idx, 700, replace = False)
        calm_Y_array = Y_array_ours[calm_idx]
        calm_X_array = X_array_ours[calm_idx]
        other_Y_array = Y_array_ours[other_idx]
        other_X_array = X_array_ours[other_idx]
        X_res_ours = np.concatenate((calm_X_array, other_X_array), axis = 0)
        Y_res_ours = np.concatenate((calm_Y_array, other_Y_array), axis = 0)
        
        X_res_ours, Y_res_ours = self.oversampling(X_res_ours, Y_res_ours, self.oversampling_method, 'not majority')
        df_ours_new = self.create_dataframe(X_res_ours, Y_res_ours)
        self.plot_histogram(participant_name = 'Ours (after oversampling)', df_new = df_ours_new)

        # sensors
        X_res_sensors, Y_res_sensors = self.undersampling(X_array_sensors, Y_array_sensors, 'not minority')
        df_sensors_new = self.create_dataframe(X_res_sensors, Y_res_sensors)
        self.plot_histogram(participant_name = 'Sensors (after undersampling)', df_new = df_sensors_new)

        # ulm-tsst
        strategy_dic = {0 : 700, 1 : 700, 2 : 700, 3 : 700}
        X_res_ulm_tsst, Y_res_ulm_tsst = self.undersampling(X_array_ulm_tsst, Y_array_ulm_tsst, strategy_dic)
        df_ulm_tsst_new = self.create_dataframe(X_res_ulm_tsst, Y_res_ulm_tsst)
        self.plot_histogram(participant_name = 'Ulm-TSST (after undersampling)', df_new = df_ulm_tsst_new)

        # all data
        X_res = np.concatenate((X_res_ours, X_res_sensors, X_res_ulm_tsst), axis = 0)
        Y_res = np.concatenate((Y_res_ours, Y_res_sensors, Y_res_ulm_tsst), axis = 0)
        df_new = self.create_dataframe(X_res, Y_res)
        self.plot_histogram(participant_name = 'Ours + Sensors + Ulm-TSST (after oversampling)', df_new = df_new)
        
        # create train and validation data
        val_size = 0.35
        X_train_ours, X_val_ours, Y_train_ours, Y_val_ours = train_test_split(X_res_ours, Y_res_ours, test_size = val_size, random_state = 42, shuffle = True, stratify = Y_res_ours)
        X_train_sensors, X_val_sensors, Y_train_sensors, Y_val_sensors = train_test_split(X_res_sensors, Y_res_sensors, test_size = val_size, random_state = 42, shuffle = True, stratify = Y_res_sensors)
        X_train_ulm_tsst, X_val_ulm_tsst, Y_train_ulm_tsst, Y_val_ulm_tsst = train_test_split(X_res_ulm_tsst, Y_res_ulm_tsst, test_size = val_size, random_state = 42, shuffle = True, stratify = Y_res_ulm_tsst)
        X_train = np.concatenate((X_train_ours, X_train_sensors, X_train_ulm_tsst), axis = 0)
        Y_train = np.concatenate((Y_train_ours, Y_train_sensors, Y_train_ulm_tsst), axis = 0)
        X_val = np.concatenate((X_val_ours, X_val_sensors, X_val_ulm_tsst), axis = 0)
        Y_val = np.concatenate((Y_val_ours, Y_val_sensors, Y_val_ulm_tsst), axis = 0)

        return (X_train, Y_train, X_val, Y_val)


    # Make calibration dataset for quantization on npu
    def make_calibration_dataset(self, X_train, save_directory, path_directory) :

        os.makedirs(save_directory, exist_ok = True)
        npy_save_path_lists = []
        X_train = X_train.astype(np.float32)
        X_train = np.expand_dims(X_train, axis = 2)  # change shape (15, 1) -> (15, 1, 1)  (*only for npu)

        for i in range(len(X_train)) :
            npy_save_path = save_directory + '/train_' + str(i) + '.npy'
            np.save(npy_save_path, X_train[i])
            npy_save_path_lists.append(npy_save_path)

        with open(path_directory, 'w') as f :
            paths_num = len(npy_save_path_lists)
            for i in range(paths_num) :
                if i == paths_num - 1 :
                    f.write(npy_save_path_lists[i])
                else :
                    f.write(npy_save_path_lists[i] + '\n')
        print('Saved calibration dataset.')
        

    # Save dataset in npy format
    def save_dataset(self, info, X_array, Y_array, save_directory) :

        X_save_directory = save_directory + '/X'
        Y_save_directory = save_directory + '/Y'
        os.makedirs(X_save_directory, exist_ok = True)
        os.makedirs(Y_save_directory, exist_ok = True)

        X_npy_save_path_lists = []
        Y_npy_save_path_lists = []

        X_array = X_array.astype(np.float32)
        Y_array = Y_array.astype(np.float32)

        X_array = np.expand_dims(X_array, axis = 2)  # change shape (15, 1) -> (15, 1, 1)  (*only for npu)
        
        for i in range(len(X_array)) :
            X_npy_save_path = X_save_directory + '/heartrate_' + str(i) + '.npy'
            np.save(X_npy_save_path, X_array[i])
            X_npy_save_path_lists.append(X_npy_save_path)

            Y_npy_save_path = Y_save_directory + '/emotion_' + str(i) + '.npy'
            np.save(Y_npy_save_path, Y_array[i])
            Y_npy_save_path_lists.append(Y_npy_save_path)
            
        X_path_directory = save_directory + '/' + info + '_X_data_paths.txt'
        with open(X_path_directory, 'w') as f :
            paths_num = len(X_npy_save_path_lists)
            for i in range(paths_num) :
                if i == paths_num - 1 :
                    f.write(X_npy_save_path_lists[i])
                else :
                    f.write(X_npy_save_path_lists[i] + '\n')

        Y_path_directory = save_directory + '/' + info + '_Y_data_paths.txt'
        with open(Y_path_directory, 'w') as f :
            paths_num = len(Y_npy_save_path_lists)
            for i in range(paths_num) :
                if i == paths_num - 1 :
                    f.write(Y_npy_save_path_lists[i])
                else :
                    f.write(Y_npy_save_path_lists[i] + '\n')

        print('Saved', info, 'dataset.')

    
    # Save dataset in npz file
    def save_dataset_one_file(self, info, X_array, Y_array, save_directory) :

        os.makedirs(save_directory, exist_ok = True)

        X_array = X_array.astype(np.float32)
        Y_array = Y_array.astype(np.float32)

        X_array = np.expand_dims(X_array, axis = 2)  # change shape (15, 1) -> (15, 1, 1)  (*only for npu)

        npz_file_name = save_directory + '/' + info + '_dataset.npz'

        np.savez(npz_file_name, x = X_array, y = Y_array)

        print('Saved', info, 'dataset.')


    # train model & evaluation
    def train_model_and_evaluation(self, df_ours_new, df_sensors_new, df_ulm_tsst_new) :

        # create train data
        X_train, Y_train, X_val, Y_val = self.create_train_data(df_ours_new, df_sensors_new, df_ulm_tsst_new)

        # train model
        X_train, Y_train = np.expand_dims(X_train, axis = 2), to_categorical(Y_train)
        print('     X_train shape :', X_train.shape, ', Y_train shape :', Y_train.shape)
        X_val, Y_val = np.expand_dims(X_val, axis = 2), to_categorical(Y_val)
        print('     X_val shape :', X_val.shape, ', Y_val shape :', Y_val.shape)

        model = self.create_model(X_train, X_val, Y_train, Y_val, 
                                self.number_of_classes, self.filters, self.kernel_sizes, 
                                self.pool_sizes, self.drop_rate, self.neurons,
                                os.path.join(self.model_dir, self.model_filename),
                                os.path.join(self.model_dir, self.history_filename), 
                                self.batch, self.number_epochs, self.modelType)

        score = self.prediction(X_val, Y_val, model, os.path.join(self.model_dir, self.pred_prefix), os.path.join(self.model_dir, self.confusion_prefix))
        print('===== Final score =====')
        print(score)

        # make calibration dataset for quantization on npu
        model_name = self.model_filename.split('.')[0]
        save_directory = self.project_dir + 'calibration/' + model_name + '/HeartRate_data'
        path_directory = self.project_dir + 'calibration/' + model_name + '/calibration_data_paths.txt'
        self.make_calibration_dataset(X_train, save_directory, path_directory)

        # save train, val dataset in npy format
        model_name = self.model_filename.split('.')[0]
        train_data_save_directory = self.project_dir + 'dataset/train_dataset/' + model_name
        val_data_save_directory = self.project_dir + 'dataset/validation_dataset/' + model_name
        self.save_dataset_one_file('train', X_train, Y_train, train_data_save_directory)
        self.save_dataset_one_file('validation', X_val, Y_val, val_data_save_directory)


    # 결과가 매번 같게 나오도록 seed 고정
    def my_seed_everywhere(self, seed: int = 42):

        tf_determinism.enable_determinism()
        os.environ["PYTHONHASHSEED"] = str(seed) # os
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        tf.random.set_seed(seed) # tensorflow
        random.seed(seed) # random
        np.random.seed(seed) # np



if __name__ == '__main__' :

    er = EmotionRecognition()

    er.project_dir                  = '/home/mines/Desktop/Projects/EmotionRecognition_HeartRate/GPU/'
    er.model_dir                    = er.project_dir + 'models/'              
    er.number_of_classes            = 4
    er.filters                      = 128                    
    er.kernel_sizes                 = [10, 6, 1, 1]          
    er.pool_sizes                   = [2, 1, 1]              
    er.drop_rate                    = 0.5                    
    er.neurons                      = [256, 128, 64]         
    er.batch                        = 32                     
    er.number_epochs                = 150                    
    er.model_filename               = 'model_-15s_0s_DCNN.hdf5'  
    er.history_filename             = 'history_-15s_0s_DCNN.csv'  
    er.pred_prefix                  = 'preds_-15s_0s_DCNN'        
    er.confusion_prefix             = 'result_conf_-15s_0s_DCNN' 
    er.modelType                    = 2                                 # select : 0(CNN_Flatten) or 1(CNN LSTM) or 2(DCNN)
    er.cluster_balance              = 0.05                   

    er.ours_emotion_file_path       = er.project_dir + 'data/EMOTION_khy+taein+mes_new.csv'
    er.ours_hr_file_path            = er.project_dir + 'data/HRDATA_khy+taein+mes_new.csv'

    er.sensors_emotion_file_path    = er.project_dir + 'data/EMOTION_sensors_new.csv'
    er.sensors_hr_file_path         = er.project_dir + 'data/HRDATA_sensors_selected_5sec_and_interpolated_1sec.csv'

    er.ult_data_file_path           = er.project_dir + 'data/c3_muse_stress_2022_total_new.csv'

    er.back_second                  = 15
    er.after_second                 = 0

    er.removing_outliers_base       = 'variance'                        # select : 'mean' or 'variance' or 'both'
    er.histogram_base               = 'mean'                            # select : 'mean' or 'variance'
    er.oversampling_method          = 'kmeans'                          # select : 'kmeans' or 'knn' or 'random'
    

    # fix seed
    er.my_seed_everywhere(42)

    # make samples [eTimeStamp - back_second <= hrTimeStamp < eTimeStamp + after_second]
    # ours data
    mos = MakeOurSamples()
    df_ours_new = mos.make_samples(er.ours_emotion_file_path, er.ours_hr_file_path, er.back_second, er.after_second, er.removing_outliers_base)

    # sensors data
    mss = MakeSensorsSamples()
    df_sensors_new = mss.make_samples(er.sensors_emotion_file_path, er.sensors_hr_file_path, er.back_second, er.after_second, er.removing_outliers_base)

    # ulm-tsst data
    mus = MakeUlmTsstSamples()
    df_ulm_tsst_new = mus.make_samples(er.ult_data_file_path, er.back_second, er.after_second, er.removing_outliers_base)

    # train model & evaluation
    er.train_model_and_evaluation(df_ours_new, df_sensors_new, df_ulm_tsst_new)
