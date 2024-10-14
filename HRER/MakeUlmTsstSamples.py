import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os



class MakeUlmTsstSamples :


    def __init__(self) :

        self.project_dir                  = '/home/mines/Desktop/Projects/EmotionRecognition_HeartRate/GPU/'
        self.data_file_path               = self.project_dir + 'data/c3_muse_stress_2022_total_new.csv'

        self.back_second                  = 15
        self.after_second                 = 0

        self.removing_outliers_base       = 'variance'   # select : 'mean' or 'variance' or 'both'
        self.histogram_base               = 'mean'       # select : 'mean' or 'variance'


    ## Min-Max Normalization
    def min_max_normalization(self, df_ulm_tsst_new) :

        # substitute the min, max value of participant-wise with the min, max value of all participants
        hr_array = df_ulm_tsst_new['hr'].to_numpy()
        hr_normalized = (hr_array - np.min(hr_array)) / (np.max(hr_array) - np.min(hr_array))
        print('min :', np.min(hr_array), ', max :', np.max(hr_array))
        df_ulm_tsst_new['hr'] = hr_normalized
        df_normalized = df_ulm_tsst_new.copy()

        return df_normalized


    ## Slicing [eTimeStamp - back_second <= hrTimeStamp < eTimeStamp + after_second]
    def slicing(self, df_normalized, imei_list, back_second, after_second) :

        df_normalized = df_normalized.replace({'neutral' : 'calm'})  # neutral -> calm
        print('emotion classes :', df_normalized['emotion'].unique())

        sample_list = []

        for imei in imei_list :
            df = df_normalized.loc[df_normalized['imei'] == imei]
            eTimeStamp_list = df['time'].tolist()
            emotion_list = df['emotion'].tolist()

            for etimestamp, emotion in zip(eTimeStamp_list, emotion_list) :
                df_sliced = df.loc[(df['time'] >= (etimestamp - back_second)) & (df['time'] < (etimestamp + after_second))]
                hr_list = df_sliced['hr'].tolist()
                # print('length :', len(hr_list))
                emotion_array = df_sliced['emotion']

                if np.all(emotion_array == emotion) :  # check if all the emotions are same
                    if len(hr_list) >= (back_second + after_second) :
                        # print('length :', len(hr_list), ', imei :', imei, ', etimestamp :', etimestamp, ', emotion :', emotion)

                        sample = []
                        sample.append(imei)
                        sample = sample + hr_list[0:back_second + after_second]

                        sample.append(emotion)

                        sample_list.append(sample)
                        # print('created sample')

        column_name = []
        column_name.append('imei')
        for i in range(0, back_second + after_second) :
            name = 'hr' + str(i)
            column_name.append(name)
        column_name.append('emotion')

        df_ulm_tsst_sample = pd.DataFrame(sample_list, columns = column_name)
        print(df_ulm_tsst_sample)

        return df_ulm_tsst_sample


    ## Make samples [eTimeStamp - back_second <= hrTimeStamp < eTimeStamp + after_second]
    def make_samples(self, data_file_path, back_second, after_second, removing_outliers_base) :

        df_ulm_tsst_new = pd.read_csv(data_file_path)

        imei_list = df_ulm_tsst_new['imei'].unique().tolist()
        print('imei :', imei_list)

        # min-max normalization
        df_normalized = self.min_max_normalization(df_ulm_tsst_new)
        print(df_normalized)

        # slicing [eTimeStamp - back_second <= hrTimeStamp < eTimeStamp + after_second]
        df_ulm_tsst_sample = self.slicing(df_normalized, imei_list, back_second, after_second)

        imei_list = df_ulm_tsst_sample['imei'].unique().tolist()
        print('imei :', imei_list)
        
        df_ulm_tsst_new = self.add_statistics_values(df_ulm_tsst_sample)
        df_new_list = []

        # participant-wise
        for imei in imei_list :
            df = df_ulm_tsst_new.loc[df_ulm_tsst_new['imei'] == imei]
            df_new = self.remove_outliers(df, removing_outliers_base)
            df_new_list.append(df_new)
        
        # all participants
        df_ulm_tsst_new = pd.concat(df_new_list, axis = 0)
        
        return df_ulm_tsst_new


    ## Add statistics values to dataframe
    def add_statistics_values(self, df_ulm_tsst_sample) :

        df_new = df_ulm_tsst_sample.copy()
        df_new['mean_hr'] = df_ulm_tsst_sample.iloc[:, 1:-1].mean(axis = 1).to_numpy()
        df_new['variance_hr'] = df_ulm_tsst_sample.iloc[:, 1:-1].var(axis = 1).to_numpy()
        # print(df_new)

        return df_new


    ## Remove outliers
    def remove_outliers(self, df, base) :

        if base == 'mean' : 
            # remove outliers based on the mean of heart rate
            Q1_mean = df['mean_hr'].quantile(q = 0.25, interpolation = 'linear')
            Q3_mean = df['mean_hr'].quantile(q = 0.75, interpolation = 'linear')
            IQR_mean = Q3_mean - Q1_mean
            print(f'\n[Based on the mean of heart rate] Q1 : {Q1_mean}, Q3 : {Q3_mean}, IQR : {IQR_mean}')

            df_removed_outliers = df.loc[(df['mean_hr'] >= Q1_mean - 1.5 * IQR_mean) & (df['mean_hr'] <= Q3_mean + 1.5 * IQR_mean)]
            # print(df_removed_outliers)

        if base == 'variance' :
            # remove outliers based on the variance of heart rate
            Q1_var = df['variance_hr'].quantile(q = 0.25, interpolation = 'linear')
            Q3_var = df['variance_hr'].quantile(q = 0.75, interpolation = 'linear')
            IQR_var = Q3_var - Q1_var
            print(f'\n[Based on the variance of heart rate] Q1 : {Q1_var}, Q3 : {Q3_var}, IQR : {IQR_var}')

            df_removed_outliers = df.loc[(df['variance_hr'] >= Q1_var - 1.5 * IQR_var) & (df['variance_hr'] <= Q3_var + 1.5 * IQR_var)]
            # print(df_removed_outliers)

        if base == 'both' :
            # remove outliers based on the mean and variance of heart rate
            Q1_mean = df['mean_hr'].quantile(q = 0.25, interpolation = 'linear')
            Q3_mean = df['mean_hr'].quantile(q = 0.75, interpolation = 'linear')
            IQR_mean = Q3_mean - Q1_mean
            print(f'\n[Based on the mean of heart rate] Q1 : {Q1_mean}, Q3 : {Q3_mean}, IQR : {IQR_mean}')

            df_removed_outliers = df.loc[(df['mean_hr'] >= Q1_mean - 1.5 * IQR_mean) & (df['mean_hr'] <= Q3_mean + 1.5 * IQR_mean)]

            Q1_var = df['variance_hr'].quantile(q = 0.25, interpolation = 'linear')
            Q3_var = df['variance_hr'].quantile(q = 0.75, interpolation = 'linear')
            IQR_var = Q3_var - Q1_var
            print(f'\n[Based on the variance of heart rate] Q1 : {Q1_var}, Q3 : {Q3_var}, IQR : {IQR_var}')

            df_removed_outliers = df_removed_outliers.loc[(df_removed_outliers['variance_hr'] >= Q1_var - 1.5 * IQR_var) & (df_removed_outliers['variance_hr'] <= Q3_var + 1.5 * IQR_var)]
            # print(df_removed_outliers)

        return df_removed_outliers


    # Plot histograms
    def plot_histogram(self, participant_name, df_new, base) :

        # plot histogram for each emotion
        print(f'-----Plot histograms for {participant_name} data-----')

        # emotion_list = df_new['emotion'].unique().tolist()
        emotion_list = ['happy', 'calm', 'sad', 'angry']

        if base == 'mean' :
            for emotion in emotion_list :
                df_emotion = df_new.loc[df_new['emotion'] == emotion]
                total_frequency = len(df_emotion)
                print(f"Total Frequency of '{emotion}' Emotion : {total_frequency}")

                plt.figure()
                plt.title(f"Mean of Heart Rate Distribution for '{emotion}' Emotion")
                plt.xlabel('Mean of Heart Rate')
                plt.ylabel('Frequency')
                sns.histplot(data = df_emotion, x = 'mean_hr', kde = True)
                # plt.show()
                file_dir = self.project_dir + 'data analysis result/MakeUlmTsstSamples.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/historgram base - ' + self.histogram_base + '/'
                os.makedirs(file_dir, exist_ok = True)
                plt.savefig(file_dir + emotion + '.png')
                plt.close()

            # plot histogram for all emotions
            plt.figure()
            plt.title(f"Mean of Heart Rate Distribution for All Emotions")
            plt.xlabel('Mean of Heart Rate')
            plt.ylabel('Frequency')
            plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            sns.histplot(data = df_new, x = 'mean_hr', kde = True, hue = 'emotion', palette = 'bright')
            # plt.show()
            file_dir = self.project_dir + 'data analysis result/MakeUlmTsstSamples.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/historgram base - ' + self.histogram_base + '/'
            os.makedirs(file_dir, exist_ok = True)
            plt.savefig(file_dir + '/all emotions.png')
            plt.close()

        if base == 'variance' :
            for emotion in emotion_list :
                df_emotion = df_new.loc[df_new['emotion'] == emotion]
                total_frequency = len(df_emotion)
                print(f"Total Frequency of '{emotion}' Emotion : {total_frequency}")

                plt.figure()
                plt.title(f"Variance of Heart Rate Distribution for '{emotion}' Emotion")
                plt.xlabel('Variance of Heart Rate')
                plt.ylabel('Frequency')
                sns.histplot(data = df_emotion, x = 'variance_hr', kde = True)
                # plt.show()
                file_dir = self.project_dir + 'data analysis result/MakeUlmTsstSamples.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/historgram base - ' + self.histogram_base + '/'
                os.makedirs(file_dir, exist_ok = True)
                plt.savefig(file_dir + emotion + '.png')
                plt.close()

            plt.figure()
            plt.title(f"Variance of Heart Rate Distribution for All Emotions")
            plt.xlabel('Variance of Heart Rate')
            plt.ylabel('Frequency')
            plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            sns.histplot(data = df_new, x = 'variance_hr', kde = True, hue = 'emotion', palette = 'bright')
            # plt.show()
            file_dir = self.project_dir + 'data analysis result/MakeUlmTsstSamples.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/historgram base - ' + self.histogram_base + '/'
            os.makedirs(file_dir, exist_ok = True)
            plt.savefig(file_dir + '/all emotions.png')
            plt.close()

        print('-----------------------------------------\n')


    # Data plot
    def data_plot(self, df_ulm_tsst_new, histogram_base) :

        imei_list = df_ulm_tsst_new['imei'].unique().tolist()
        print('imei :', imei_list)

        # participant-wise
        for imei in imei_list :
            df_new = df_ulm_tsst_new.loc[df_ulm_tsst_new['imei'] == imei]
            # self.plot_histogram(participant_name = str(imei), df_new = df_new, base = histogram_base)

            # print the number of the participant's data
            print(f'-----Participant {imei} data-----')
            emotion_list = ['happy', 'calm', 'sad', 'angry']
            for emotion in emotion_list :
                df_emotion = df_new.loc[df_new['emotion'] == emotion]
                total_frequency = len(df_emotion)
                print(f"Total Frequency of '{emotion}' Emotion : {total_frequency}")
        
        # all participants
        self.plot_histogram(participant_name = 'All participants', df_new = df_ulm_tsst_new, base = histogram_base)
   
    

if __name__ == '__main__' :

    ms = MakeUlmTsstSamples()    

    ms.project_dir                  = '/home/mines/Desktop/Projects/EmotionRecognition_HeartRate/GPU/'
    ms.data_file_path               = ms.project_dir + 'data/c3_muse_stress_2022_total_new.csv'

    ms.back_second                  = 5
    ms.after_second                 = 0

    ms.removing_outliers_base       = 'variance'   # select : 'mean' or 'variance' or 'both'
    ms.histogram_base               = 'mean'       # select : 'mean' or 'variance'

    df_ulm_tsst_new = ms.make_samples(ms.data_file_path, ms.back_second, ms.after_second, ms.removing_outliers_base)
    ms.data_plot(df_ulm_tsst_new, ms.histogram_base)