import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os



class MakeOurSamples :


    def __init__(self) :

        self.project_dir                  = '/home/mines/Desktop/Projects/EmotionRecognition_HeartRate/GPU/'
        self.emotion_file_path            = self.project_dir + 'data/EMOTION_khy+taein+mes_new.csv'
        self.hr_file_path                 = self.project_dir + 'data/HRDATA_khy+taein+mes_new.csv'

        self.back_second                  = 15
        self.after_second                 = 0

        self.removing_outliers_base       = 'variance'  # select : 'mean' or 'variance' or 'both'
        self.histogram_base               = 'mean'      # select : 'mean' or 'variance'


    ## Min-Max Normalization
    def min_max_normalization(self, df_ours_hr_new, participant_number) :

        if participant_number == 1 : # hoyeon
            df = df_ours_hr_new.loc[(df_ours_hr_new['imei'] > 1) & (df_ours_hr_new['imei'] < 2)]
        
        if participant_number == 2 : # taein
            df = df_ours_hr_new.loc[df_ours_hr_new['imei'] == 2]

        if participant_number == 3 : # eesun
            df = df_ours_hr_new.loc[(df_ours_hr_new['imei'] > 3) & (df_ours_hr_new['imei'] < 4)]

        hr_array = df['mean_hr'].to_numpy()
        hr_normalized = (hr_array - np.min(hr_array)) / (np.max(hr_array) - np.min(hr_array))
        # print('min :', np.min(hr_array), ', max :', np.max(hr_array))
        df['mean_hr'] = hr_normalized

        return df


    ## Slicing [eTimeStamp - back_second <= hrTimeStamp < eTimeStamp + after_second]
    def slicing(self, df_ours_emotion_new, df_hr_normalized, back_second, after_second) :

        df_ours_emotion_new = df_ours_emotion_new.replace({'neutral' : 'calm'})  # neutral -> calm
        print('emotion classes :', df_ours_emotion_new['mode_emotion'].unique())

        imei_list = df_ours_emotion_new['deviceID'].unique().tolist()
        print('imei :', imei_list)

        sample_list = []

        for imei in imei_list :
            df_emotion = df_ours_emotion_new.loc[df_ours_emotion_new['deviceID'] == imei]
            eTimeStamp_list = df_emotion['eTimeStamp'].tolist()
            emotion_list = df_emotion['mode_emotion'].tolist()

            df_hr = df_hr_normalized.loc[df_hr_normalized['imei'] == imei]
            last_etimestamp = None
            previous_emotion = None

            for i in range(len(emotion_list)) :
                current_etimestamp = eTimeStamp_list[i]
                current_emotion = emotion_list[i]

                if len(emotion_list) < 2 :
                    enough_backspace = True
                    enough_afterspace = True

                else :
                    if i == 0 :
                        next_etimestamp = eTimeStamp_list[i + 1]
                        afterspace = next_etimestamp - current_etimestamp
                        enough_backspace = True
                        enough_afterspace = (afterspace >= after_second)
                        is_dominant_backspace = True
                        is_dominant_afterspace = (afterspace >= np.ceil(after_second / 2))
                        if last_etimestamp is not None :
                            is_dominant_lastspace = True

                    elif i == (len(emotion_list) - 1) :
                        previous_etimestamp = eTimeStamp_list[i - 1]
                        previous_emotion = emotion_list[i - 1]
                        backspace = current_etimestamp - previous_etimestamp
                        enough_backspace = (backspace >=  back_second)
                        enough_afterspace = True
                        is_dominant_backspace = (backspace >= np.ceil(back_second / 2))
                        is_dominant_afterspace = True
                        if last_etimestamp is not None :
                            lastspace = current_etimestamp - last_etimestamp
                            is_dominant_lastspace = (lastspace >= np.ceil(back_second / 2))

                    else :
                        previous_etimestamp = eTimeStamp_list[i - 1]
                        previous_emotion = emotion_list[i - 1]
                        next_etimestamp = eTimeStamp_list[i + 1]
                        backspace = current_etimestamp - previous_etimestamp
                        afterspace = next_etimestamp - current_etimestamp
                        enough_backspace = (backspace >=  back_second)
                        enough_afterspace = (afterspace >= after_second)
                        is_dominant_backspace = (backspace >= np.ceil(back_second / 2))
                        is_dominant_afterspace = (afterspace >= np.ceil(after_second / 2))
                        if last_etimestamp is not None :
                            lastspace = current_etimestamp - last_etimestamp
                            is_dominant_lastspace = (lastspace >= np.ceil(back_second / 2))

                if enough_backspace and enough_afterspace : # Does the current emotion have enough time space?
                    df_hr_sliced = df_hr.loc[(df_hr['time'] >= (current_etimestamp - back_second)) & (df_hr['time'] < (current_etimestamp + after_second))]
                    hr_list = df_hr_sliced['mean_hr'].tolist()

                    if len(hr_list) >= (back_second + after_second) :
                        sample = []
                        sample.append(imei)
                        sample = sample + hr_list[0:back_second + after_second]
                        sample.append(current_emotion) # slicing as current emotion
                        sample_list.append(sample)
                        # print('created sample')

                else :
                    if (previous_emotion is not None) and (previous_emotion != current_emotion) : # Is the current emotion same as the previous emotion?
                        pass # no slicing
                        last_etimestamp = previous_etimestamp # the last emotion's timestamp

                    else :
                        if is_dominant_backspace and is_dominant_afterspace : # Is the current emotion dominant emotion?
                            df_hr_sliced = df_hr.loc[(df_hr['time'] >= (current_etimestamp - back_second)) & (df_hr['time'] < (current_etimestamp + after_second))]
                            hr_list = df_hr_sliced['mean_hr'].tolist()

                            if len(hr_list) >= (back_second + after_second) :
                                sample = []
                                sample.append(imei)
                                sample = sample + hr_list[0:back_second + after_second]
                                sample.append(current_emotion) # slicing as current emotion
                                sample_list.append(sample)
                                # print('created sample')

                        else :
                            if last_etimestamp is not None :
                                if (previous_emotion is not None) and is_dominant_lastspace : # Does the current emotion have enough time space from the last emotion?
                                    df_hr_sliced = df_hr.loc[(df_hr['time'] >= (current_etimestamp - back_second)) & (df_hr['time'] < (current_etimestamp + after_second))]
                                    hr_list = df_hr_sliced['mean_hr'].tolist()

                                    if len(hr_list) >= (back_second + after_second) :
                                        sample = []
                                        sample.append(imei)
                                        sample = sample + hr_list[0:back_second + after_second]
                                        sample.append(previous_emotion) # slicing as previous emotion
                                        sample_list.append(sample)
                                        # print('created sample')

                                else :
                                    pass # no slicing
                            
                            else :
                                if previous_emotion is not None :
                                    df_hr_sliced = df_hr.loc[(df_hr['time'] >= (current_etimestamp - back_second)) & (df_hr['time'] < (current_etimestamp + after_second))]
                                    hr_list = df_hr_sliced['mean_hr'].tolist()
                                    
                                    if len(hr_list) >= (back_second + after_second) :
                                        sample = []
                                        sample.append(imei)
                                        sample = sample + hr_list[0:back_second + after_second]
                                        sample.append(previous_emotion) # slicing as previous emotion
                                        sample_list.append(sample)
                                        # print('created sample')

        column_name = []
        column_name.append('imei')
        for i in range(0, back_second + after_second) :
            name = 'hr' + str(i)
            column_name.append(name)
        column_name.append('emotion')

        df_ours_sample = pd.DataFrame(sample_list, columns = column_name)
        print(df_ours_sample)

        return df_ours_sample


    ## Make samples [eTimeStamp - back_second <= hrTimeStamp < eTimeStamp + after_second]
    def make_samples(self, emotion_file_path, hr_file_path, back_second, after_second, removing_outliers_base) :

        df_ours_emotion_new = pd.read_csv(emotion_file_path)
        df_ours_hr_new = pd.read_csv(hr_file_path)

        # min-max normalization ---------------------------------------------------------------
        imei_list = df_ours_hr_new['imei'].unique().tolist()
        print('imei :', imei_list)

        # hoyeon
        df_khy = self.min_max_normalization(df_ours_hr_new, participant_number = 1)
        
        # taein
        df_taein = self.min_max_normalization(df_ours_hr_new, participant_number = 2)

        # eesun
        df_mes = self.min_max_normalization(df_ours_hr_new, participant_number = 3)

        # concatenate dataframe
        df_hr_normalized = pd.concat([df_khy, df_taein, df_mes], axis = 0)
        print(df_hr_normalized)
        # --------------------------------------------------------------------------------------

        # slicing [eTimeStamp - back_second <= hrTimeStamp < eTimeStamp + after_second]
        df_ours_sample = self.slicing(df_ours_emotion_new, df_hr_normalized, back_second, after_second)

        # hoyeon
        df_khy_new = self.add_statistics_values(df_ours_sample, participant_number = 1)
        df_khy_new = self.remove_outliers(df_khy_new, removing_outliers_base)
        
        # taein 
        df_taein_new = self.add_statistics_values(df_ours_sample, participant_number = 2)
        df_taein_new = self.remove_outliers(df_taein_new, removing_outliers_base)
    
        # eesun
        df_mes_new = self.add_statistics_values(df_ours_sample, participant_number = 3)
        df_mes_new = self.remove_outliers(df_mes_new, removing_outliers_base)

        # all participants
        df_ours_new = pd.concat([df_khy_new, df_taein_new, df_mes_new], axis = 0)

        
        return df_ours_new


    ## Add statistics values to dataframe
    def add_statistics_values(self, df_ours_sample, participant_number) :

        if participant_number == 1 : # hoyeon
            df = df_ours_sample.loc[(df_ours_sample['imei'] > 1) & (df_ours_sample['imei'] < 2)]
        
        if participant_number == 2 : # taein
            df = df_ours_sample.loc[df_ours_sample['imei'] == 2]

        if participant_number == 3 : # eesun
            df = df_ours_sample.loc[(df_ours_sample['imei'] > 3) & (df_ours_sample['imei'] < 4)]

        df_new = df.copy()
        df_new['mean_hr'] = df.iloc[:, 1:-1].mean(axis = 1).to_numpy()
        df_new['variance_hr'] = df.iloc[:, 1:-1].var(axis = 1).to_numpy()
        # print(df_new)

        return df_new


    ## Remove outliers
    def remove_outliers(self, df_new, base) :

        if base == 'mean' : 
            # remove outliers based on the mean of heart rate
            Q1_mean = df_new['mean_hr'].quantile(q = 0.25, interpolation = 'linear')
            Q3_mean = df_new['mean_hr'].quantile(q = 0.75, interpolation = 'linear')
            IQR_mean = Q3_mean - Q1_mean
            print(f'\n[Based on the mean of heart rate] Q1 : {Q1_mean}, Q3 : {Q3_mean}, IQR : {IQR_mean}')

            df_removed_outliers = df_new.loc[(df_new['mean_hr'] >= Q1_mean - 1.5 * IQR_mean) & (df_new['mean_hr'] <= Q3_mean + 1.5 * IQR_mean)]
            # print(df_removed_outliers)

        if base == 'variance' :
            # remove outliers based on the variance of heart rate
            Q1_var = df_new['variance_hr'].quantile(q = 0.25, interpolation = 'linear')
            Q3_var = df_new['variance_hr'].quantile(q = 0.75, interpolation = 'linear')
            IQR_var = Q3_var - Q1_var
            print(f'\n[Based on the variance of heart rate] Q1 : {Q1_var}, Q3 : {Q3_var}, IQR : {IQR_var}')

            df_removed_outliers = df_new.loc[(df_new['variance_hr'] >= Q1_var - 1.5 * IQR_var) & (df_new['variance_hr'] <= Q3_var + 1.5 * IQR_var)]
            # print(df_removed_outliers)

        if base == 'both' :
            # remove outliers based on the mean and variance of heart rate
            Q1_mean = df_new['mean_hr'].quantile(q = 0.25, interpolation = 'linear')
            Q3_mean = df_new['mean_hr'].quantile(q = 0.75, interpolation = 'linear')
            IQR_mean = Q3_mean - Q1_mean
            print(f'\n[Based on the mean of heart rate] Q1 : {Q1_mean}, Q3 : {Q3_mean}, IQR : {IQR_mean}')

            df_removed_outliers = df_new.loc[(df_new['mean_hr'] >= Q1_mean - 1.5 * IQR_mean) & (df_new['mean_hr'] <= Q3_mean + 1.5 * IQR_mean)]

            Q1_var = df_new['variance_hr'].quantile(q = 0.25, interpolation = 'linear')
            Q3_var = df_new['variance_hr'].quantile(q = 0.75, interpolation = 'linear')
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
                file_dir = self.project_dir + 'data analysis result/MakeOurSamples.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/historgram base - ' + self.histogram_base + '/'
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
            file_dir = self.project_dir + 'data analysis result/MakeOurSamples.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/historgram base - ' + self.histogram_base + '/'
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
                file_dir = self.project_dir + 'data analysis result/MakeOurSamples.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/historgram base - ' + self.histogram_base + '/'
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
            file_dir = self.project_dir + 'data analysis result/MakeOurSamples.py/'+ participant_name + '/-' + str(self.back_second) + 's ~ +' + str(self.after_second) + 's/removed outliers by ' + self.removing_outliers_base + '/historgram base - ' + self.histogram_base + '/'
            os.makedirs(file_dir, exist_ok = True)
            plt.savefig(file_dir + '/all emotions.png')
            plt.close()

        print('-----------------------------------------\n')


    # Data plot
    def data_plot(self, df_ours_new, histogram_base) :

        # hoyeon
        df_khy_new = df_ours_new.loc[(df_ours_new['imei'] > 1) & (df_ours_new['imei'] < 2)]
        self.plot_histogram(participant_name = 'Hoyeon', df_new = df_khy_new, base = histogram_base)
        
        # taein 
        df_taein_new = df_ours_new.loc[df_ours_new['imei'] == 2]
        self.plot_histogram(participant_name = 'Taein', df_new = df_taein_new, base = histogram_base)
    
        # eesun
        df_mes_new = df_ours_new.loc[(df_ours_new['imei'] > 3) & (df_ours_new['imei'] < 4)]
        self.plot_histogram(participant_name = 'Eesun', df_new = df_mes_new, base = histogram_base)

        # all participants
        self.plot_histogram(participant_name = 'All participants', df_new = df_ours_new, base = histogram_base)

   

if __name__ == '__main__' :

    ms = MakeOurSamples()

    ms.project_dir                  = '/home/mines/Desktop/Projects/EmotionRecognition_HeartRate/GPU/'
    ms.emotion_file_path            = ms.project_dir + 'data/EMOTION_khy+taein+mes_new.csv'
    ms.hr_file_path                 = ms.project_dir + 'data/HRDATA_khy+taein+mes_new.csv'

    ms.back_second                  = 15
    ms.after_second                 = 0

    ms.removing_outliers_base       = 'variance'   # select : 'mean' or 'variance' or 'both'
    ms.histogram_base               = 'mean'       # select : 'mean' or 'variance'

    df_ours_new = ms.make_samples(ms.emotion_file_path, ms.hr_file_path, ms.back_second, ms.after_second, ms.removing_outliers_base)
    ms.data_plot(df_ours_new, ms.histogram_base)