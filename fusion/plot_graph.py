import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import threading
import time
from keras.models import load_model
from get_resource_usage import GPU



class Plot :


    def __init__(self, gpu_device_id) :

        self.gpu = GPU(gpu_device_id)

        self.project_dir = '/home/mines/Desktop/Projects/Multimodal_Fusion/scorebased_fusion'
        self.graph_dir = self.project_dir + '/graph'
        self.model_dir = self.project_dir + '/model/'
        self.data_dir = self.project_dir + '/data'
        self.modal_dic = {
                            'HR' : 
                                    {
                                        'deep_model_name' : 'HR_model_deep', 
                                        'shallow_model_name' : 'HR_model_shallow',
                                        'deep_model_size' : 1378520,    # bytes
                                        'shallow_model_size' : 842640,  # bytes
                                        'deep_model_params_num' : 109540,
                                        'shallow_model_params_num' : 67204
                                    },

                            'EEG' : 
                                    {
                                        'deep_model_name' : 'EEG_model_deep', 
                                        'shallow_model_name' : 'EEG_model_shallow',
                                        'deep_model_size' : 935672,    # bytes
                                        'shallow_model_size' : 421864,  # bytes
                                        'deep_model_params_num' : 71804,
                                        'shallow_model_params_num' : 31404
                                    },

                            'Speech' : 
                                    {
                                        'deep_model_name' : 'Speech_model_deep', 
                                        'shallow_model_name' : 'Speech_model_shallow',
                                        'deep_model_size' : 86386384,    # bytes
                                        'shallow_model_size' : 59656704,  # bytes 
                                        'deep_model_params_num' : 7191684,
                                        'shallow_model_params_num' : 4967172
                                    },

                            'Video' : 
                                    {
                                        'deep_model_name' : 'Video_model_deep', 
                                        'shallow_model_name' : 'Video_model_shallow',
                                        'deep_model_size' : 87149344,    # bytes
                                        'shallow_model_size' : 48717720,  # bytes 
                                        'deep_model_params_num' : 7255236,
                                        'shallow_model_params_num' : 4055236
                                    },

                            'Multimodal' : 
                                    {
                                        'data_dir' : self.data_dir + '/Multimodal/',
                                        'data_name' : 'multimodal_dataset'
                                    }
                         }
        
        self.model_suffix = '.hdf5'
        self.data_suffix = '.npz'
        
    
    def load_model(self) :

        '''Load models.'''

        HR_deep_model_path = self.model_dir + self.modal_dic['HR']['deep_model_name'] + self.model_suffix
        EEG_deep_model_path = self.model_dir + self.modal_dic['EEG']['deep_model_name'] + self.model_suffix
        Speech_deep_model_path = self.model_dir + self.modal_dic['Speech']['deep_model_name'] + self.model_suffix
        Video_deep_model_path = self.model_dir + self.modal_dic['Video']['deep_model_name'] + self.model_suffix

        HR_shallow_model_path = self.model_dir + self.modal_dic['HR']['shallow_model_name'] + self.model_suffix
        EEG_shallow_model_path = self.model_dir + self.modal_dic['EEG']['shallow_model_name'] + self.model_suffix
        Speech_shallow_model_path = self.model_dir + self.modal_dic['Speech']['shallow_model_name'] + self.model_suffix
        Video_shallow_model_path = self.model_dir + self.modal_dic['Video']['shallow_model_name'] + self.model_suffix

        self.HR_deep_model = load_model(HR_deep_model_path)
        self.EEG_deep_model = load_model(EEG_deep_model_path)
        self.Speech_deep_model = load_model(Speech_deep_model_path)
        self.Video_deep_model = load_model(Video_deep_model_path)

        self.HR_shallow_model = load_model(HR_shallow_model_path)
        self.EEG_shallow_model = load_model(EEG_shallow_model_path)
        self.Speech_shallow_model = load_model(Speech_shallow_model_path)
        self.Video_shallow_model = load_model(Video_shallow_model_path)

        print('Loaded models.')


    def load_data(self) :

        '''Load multimodal data.'''

        multimodal_data = np.load(self.modal_dic['Multimodal']['data_dir'] + self.modal_dic['Multimodal']['data_name'] + self.data_suffix)
        hr_x = multimodal_data['hr_x']
        eeg_x = multimodal_data['eeg_x']
        speech_x = multimodal_data['speech_x']
        video_x = multimodal_data['video_x']

        print('hr_x shape :', hr_x.shape)
        print('eeg_x shape :', eeg_x.shape)
        print('speech_x shape :', speech_x.shape)
        print('video_x shape :', video_x.shape)

        print('Loaded multimodal data.')

        return hr_x, eeg_x, speech_x, video_x


    def prediction(self, mode, model_depth, model_num, hr_x, eeg_x, speech_x, video_x):

        '''
        Model Prediction.

        Parameters
            - mode 
                - 'by one' : predict by one data
                - 'only one' : predict only the first data
            - model_depth
                - 'deep' : deep model
                - 'shallow' : shallow model 
            - model_num : model number (from 1 to the 4)
        '''

        self.gpu.clean_buffer()
        self.gpu.event.clear()
        t = threading.Thread(target = self.gpu.get_GPU_Usage)
        t.start()

        if mode == 'by one' :
            if model_depth == 'deep' :
                for h_x, e_x, s_x, v_x in zip(hr_x, eeg_x, speech_x, video_x) :
                    h_x = np.expand_dims(h_x, axis = 0)
                    e_x = np.expand_dims(e_x, axis = 0)
                    s_x = np.expand_dims(s_x, axis = 0)
                    v_x = np.expand_dims(v_x, axis = 0)

                    _ = self.HR_deep_model.predict(h_x, verbose = 0)
                    _ = self.EEG_deep_model.predict(e_x)
                    _ = self.Speech_deep_model.predict(s_x)
                    _ = self.Video_deep_model.predict(v_x)

            if model_depth == 'shallow' :
                for h_x, e_x, s_x, v_x in zip(hr_x, eeg_x, speech_x, video_x) :
                    h_x = np.expand_dims(h_x, axis = 0)
                    e_x = np.expand_dims(e_x, axis = 0)
                    s_x = np.expand_dims(s_x, axis = 0)
                    v_x = np.expand_dims(v_x, axis = 0)

                    _ = self.HR_shallow_model.predict(h_x, verbose = 0)
                    _ = self.EEG_shallow_model.predict(e_x)
                    _ = self.Speech_shallow_model.predict(s_x)
                    _ = self.Video_shallow_model.predict(v_x)

            self.gpu.event.set()
            avg_gpu_util, avg_memory_util = self.gpu.calculate_GPU_average_Usage()
            print(f'Average GPU Util : {avg_gpu_util}%, Average Memory Util : {avg_memory_util}%')

            return self.gpu.gpu_util_buffer, self.gpu.memory_util_buffer, avg_gpu_util, avg_memory_util
        

        if mode == 'only one' :
            if model_depth == 'deep' :
                if model_num == 1 :
                    h_x = np.expand_dims(hr_x[0], axis = 0)
                    _ = self.HR_deep_model.predict(h_x, verbose = 0)

                if model_num == 2 :
                    h_x = np.expand_dims(hr_x[0], axis = 0)
                    e_x = np.expand_dims(eeg_x[0], axis = 0)
                    _ = self.HR_deep_model.predict(h_x, verbose = 0)
                    _ = self.EEG_deep_model.predict(e_x)

                if model_num == 3 :
                    h_x = np.expand_dims(hr_x[0], axis = 0)
                    e_x = np.expand_dims(eeg_x[0], axis = 0)
                    s_x = np.expand_dims(speech_x[0], axis = 0)
                    _ = self.HR_deep_model.predict(h_x, verbose = 0)
                    _ = self.EEG_deep_model.predict(e_x)
                    _ = self.Speech_deep_model.predict(s_x)

                if model_num == 4 :
                    h_x = np.expand_dims(hr_x[0], axis = 0)
                    e_x = np.expand_dims(eeg_x[0], axis = 0)
                    s_x = np.expand_dims(speech_x[0], axis = 0)
                    v_x = np.expand_dims(video_x[0], axis = 0)
                    _ = self.HR_deep_model.predict(h_x, verbose = 0)
                    _ = self.EEG_deep_model.predict(e_x)
                    _ = self.Speech_deep_model.predict(s_x)
                    _ = self.Video_deep_model.predict(v_x)
                
            if model_depth == 'shallow' :
                if model_num == 1 :
                    h_x = np.expand_dims(hr_x[0], axis = 0)
                    _ = self.HR_shallow_model.predict(h_x, verbose = 0)

                if model_num == 2 :
                    h_x = np.expand_dims(hr_x[0], axis = 0)
                    e_x = np.expand_dims(eeg_x[0], axis = 0)
                    _ = self.HR_shallow_model.predict(h_x, verbose = 0)
                    _ = self.EEG_shallow_model.predict(e_x)
                   
                if model_num == 3 :
                    h_x = np.expand_dims(hr_x[0], axis = 0)
                    e_x = np.expand_dims(eeg_x[0], axis = 0)
                    s_x = np.expand_dims(speech_x[0], axis = 0)
                    _ = self.HR_shallow_model.predict(h_x, verbose = 0)
                    _ = self.EEG_shallow_model.predict(e_x)
                    _ = self.Speech_shallow_model.predict(s_x)

                if model_num == 4 :
                    h_x = np.expand_dims(hr_x[0], axis = 0)
                    e_x = np.expand_dims(eeg_x[0], axis = 0)
                    s_x = np.expand_dims(speech_x[0], axis = 0)
                    v_x = np.expand_dims(video_x[0], axis = 0)
                    _ = self.HR_shallow_model.predict(h_x, verbose = 0)
                    _ = self.EEG_shallow_model.predict(e_x)
                    _ = self.Speech_shallow_model.predict(s_x)
                    _ = self.Video_shallow_model.predict(v_x)

            self.gpu.event.set()
            avg_gpu_util, avg_memory_util = self.gpu.calculate_GPU_average_Usage()
            print(f'Average GPU Util : {avg_gpu_util}%, Average Memory Util : {avg_memory_util}%')

            return avg_gpu_util, avg_memory_util


    def get_model_information(self, modal) :

        '''Get model size and total parameter number.'''

        deep_model_size = self.modal_dic[modal]['deep_model_size']
        shallow_model_size = self.modal_dic[modal]['shallow_model_size']
        deep_model_prams_num = self.modal_dic[modal]['deep_model_params_num']
        shallow_model_prams_num = self.modal_dic[modal]['shallow_model_params_num']

        return deep_model_size, shallow_model_size, deep_model_prams_num, shallow_model_prams_num


    def ModelSize_ParameterNumber_graph(self) :

        '''
        Plot a graph.

        x-axis : Individual model (HR, EEG, Speech, Video)
        y-axis : Model size and Total parameter number of model
        '''

        HR_deep_model_size, HR_shallow_model_size, HR_deep_model_prams_num, HR_shallow_model_prams_num = self.get_model_information('HR')
        EEG_deep_model_size, EEG_shallow_model_size, EEG_deep_model_prams_num, EEG_shallow_model_prams_num = self.get_model_information('EEG')
        Speech_deep_model_size, Speech_shallow_model_size, Speech_deep_model_prams_num, Speech_shallow_model_prams_num = self.get_model_information('Speech')
        Video_deep_model_size, Video_shallow_model_size, Video_deep_model_prams_num, Video_shallow_model_prams_num = self.get_model_information('Video')

        df = pd.DataFrame({'Individual Model' : ['Heart-Rate', 'Heart-Rate', 
                                                 'EEG', 'EEG', 
                                                 'Speech', 'Speech', 
                                                 'Video', 'Video'],

                           'Model Depth' : ['Deep', 'Shallow', 
                                            'Deep', 'Shallow', 
                                            'Deep', 'Shallow', 
                                            'Deep', 'Shallow'],

                           'Model Size (Bytes)' : [HR_deep_model_size, HR_shallow_model_size, 
                                           EEG_deep_model_size, EEG_shallow_model_size, 
                                           Speech_deep_model_size, Speech_shallow_model_size, 
                                           Video_deep_model_size, Video_shallow_model_size],

                           'Total Parameters Number' : [HR_deep_model_prams_num, HR_shallow_model_prams_num,
                                                        EEG_deep_model_prams_num, EEG_shallow_model_prams_num,
                                                        Speech_deep_model_prams_num, Speech_shallow_model_prams_num,
                                                        Video_deep_model_prams_num, Video_shallow_model_prams_num]})
        print(df)

        sns.set_style('whitegrid')
        sns.set_palette('muted')

        # plot Model size graph
        model_size_graph_path = self.graph_dir + '/ModelSize_graph.png'
        sns.barplot(x = 'Individual Model', y = 'Model Size (Bytes)', hue = 'Model Depth', data = df)
        plt.savefig(model_size_graph_path)
        plt.close()

        # plot Total Parameter number graph
        params_num_graph_path = self.graph_dir + '/TotalParameterNumber_graph.png'
        sns.barplot(x = 'Individual Model', y = 'Total Parameters Number', hue = 'Model Depth', data = df)
        plt.savefig(params_num_graph_path)
        plt.close()


    def ModelNumber_GPUusage_graph(self) :

        '''
        Plot a graph.

        x-axis : Individual Model number
        y-axis : Average GPU Usage of model (GPU Util / Memory Util)
        '''

        self.load_model()
        hr_x, eeg_x, speech_x, video_x = self.load_data()
        deep1_avg_gpu_util, deep1_avg_memory_util = self.prediction('only one', 'deep', 1, hr_x, eeg_x, speech_x, video_x)
        time.sleep(10)
        deep2_avg_gpu_util, deep2_avg_memory_util = self.prediction('only one', 'deep', 2, hr_x, eeg_x, speech_x, video_x)
        time.sleep(10)
        deep3_avg_gpu_util, deep3_avg_memory_util = self.prediction('only one', 'deep', 3, hr_x, eeg_x, speech_x, video_x)
        time.sleep(10)
        deep4_avg_gpu_util, deep4_avg_memory_util = self.prediction('only one', 'deep', 4, hr_x, eeg_x, speech_x, video_x)
        time.sleep(10)

        shallow1_avg_gpu_util, shallow1_avg_memory_util = self.prediction('only one', 'shallow', 1, hr_x, eeg_x, speech_x, video_x)
        time.sleep(10)
        shallow2_avg_gpu_util, shallow2_avg_memory_util = self.prediction('only one', 'shallow', 2, hr_x, eeg_x, speech_x, video_x)
        time.sleep(10)
        shallow3_avg_gpu_util, shallow3_avg_memory_util = self.prediction('only one', 'shallow', 3, hr_x, eeg_x, speech_x, video_x)
        time.sleep(10)
        shallow4_avg_gpu_util, shallow4_avg_memory_util = self.prediction('only one', 'shallow', 4, hr_x, eeg_x, speech_x, video_x)

        df = pd.DataFrame({'Individual Model Number' : [1, 1,
                                                        2, 2,
                                                        3, 3,
                                                        4, 4],

                           'Model Depth' : ['Deep', 'Shallow', 
                                            'Deep', 'Shallow', 
                                            'Deep', 'Shallow', 
                                            'Deep', 'Shallow'],

                           'GPU Utilization (%)' : [deep1_avg_gpu_util, shallow1_avg_gpu_util, 
                                                deep2_avg_gpu_util, shallow2_avg_gpu_util, 
                                                deep3_avg_gpu_util, shallow3_avg_gpu_util, 
                                                deep4_avg_gpu_util, shallow4_avg_gpu_util],

                           'Memory Utilization (%)' : [deep1_avg_memory_util, shallow1_avg_memory_util,
                                                   deep2_avg_memory_util, shallow2_avg_memory_util,
                                                   deep3_avg_memory_util, shallow3_avg_memory_util,
                                                   deep4_avg_memory_util, shallow4_avg_memory_util]})
        print(df)

        sns.set_style('whitegrid')
        sns.set_palette('muted')

        # plot GPU-Util graph
        gpuUtil_graph_path = self.graph_dir + '/ModelNumber_GPU-Util_graph.png'
        plt.xticks([1, 2, 3, 4])
        sns.lineplot(x = 'Individual Model Number', y = 'GPU Utilization (%)', hue = 'Model Depth', data = df)
        plt.savefig(gpuUtil_graph_path)
        plt.close()

        # plot Memory-Util graph
        memoryUtil_graph_path = self.graph_dir + '/ModelNumber_Memory-Util_graph.png'
        plt.xticks([1, 2, 3, 4])
        sns.lineplot(x = 'Individual Model Number', y = 'Memory Utilization (%)', hue = 'Model Depth', data = df)
        plt.savefig(memoryUtil_graph_path)
        plt.close()


    def Data_GPUusage_graph(self) :

        '''
        Plot a graph.

        x-axis : Multimodal Data Index
        y-axis : GPU Usage of Fusion model (GPU Util / Memory Util)
        '''

        self.load_model()
        hr_x, eeg_x, speech_x, video_x = self.load_data()
        # data_index = [i for i in range(len(hr_x))]
        # deep_index = ['Deep' for _ in range(len(hr_x))]
        # shallow_index = ['Shallow' for _ in range(len(hr_x))]
        deep_gpu_util_buffer, deep_memory_util_buffer, deep_avg_gpu_util, deep_avg_memory_util = self.prediction('by one', 'deep', 0, hr_x, eeg_x, speech_x, video_x)
        print('**** Deep model prediction done. ****')
        time.sleep(10)
        shallow_gpu_util_buffer, shallow_memory_util_buffer, shallow_avg_gpu_util, shallow_avg_memory_util = self.prediction('by one', 'shallow', 0, hr_x, eeg_x, speech_x, video_x)
        print('**** Shallow model prediction done. ****')

        '''df = pd.DataFrame({'Multimodal Data Index' : data_index * 2,

                           'Model Depth' : deep_index + shallow_index,

                           'GPU Utilization of Fusion Model (%)' : deep_gpu_util_buffer + shallow_gpu_util_buffer,

                           'Memory Utilization of Fusion Model (%)' : deep_memory_util_buffer + shallow_memory_util_buffer})'''
        

        deep_gpu_util_buffer = list(np.array(deep_gpu_util_buffer) * 100)
        shallow_gpu_util_buffer = list(np.array(shallow_gpu_util_buffer) * 100)
        deep_memory_util_buffer = list(np.array(deep_memory_util_buffer) * 100)
        shallow_memory_util_buffer = list(np.array(shallow_memory_util_buffer) * 100)

        deep_time_index = [i for i in range(len(deep_gpu_util_buffer))]
        shallow_time_index = [i for i in range(len(shallow_gpu_util_buffer))]
        deep_index = ['Deep' for _ in range(len(deep_gpu_util_buffer))]
        shallow_index = ['Shallow' for _ in range(len(shallow_gpu_util_buffer))]

        df = pd.DataFrame({'Time (sec)' : deep_time_index + shallow_time_index,

                           'Model Depth' : deep_index + shallow_index,

                           'GPU Utilization of Fusion Model (%)' : deep_gpu_util_buffer + shallow_gpu_util_buffer,

                           'Memory Utilization of Fusion Model (%)' : deep_memory_util_buffer + shallow_memory_util_buffer})

        print(df)
        print(f'Averagae GPU Utilization and Memory Utilization of Deep Fusion Model (%) : {deep_avg_gpu_util}, {deep_avg_memory_util}')
        print(f'Averagae GPU Utilization and Memory Utilization of Shallow Fusion Model (%) : {shallow_avg_gpu_util}, {shallow_avg_memory_util}')

        sns.set_style('whitegrid')
        sns.set_palette('muted')

        # plot GPU-Util graph
        gpuUtil_graph_path = self.graph_dir + '/Data_GPU-Util_graph.png'
        sns.lineplot(x = 'Time (sec)', y = 'GPU Utilization of Fusion Model (%)', hue = 'Model Depth', data = df)
        plt.savefig(gpuUtil_graph_path)
        plt.close()

        # plot Memory-Util graph
        memoryUtil_graph_path = self.graph_dir + '/Data_Memory-Util_graph.png'
        sns.lineplot(x = 'Time (sec)', y = 'Memory Utilization of Fusion Model (%)', hue = 'Model Depth', data = df)
        plt.savefig(memoryUtil_graph_path)
        plt.close()



if __name__ == '__main__' :

    plot = Plot(gpu_device_id = 0)

    os.makedirs(plot.graph_dir, exist_ok = True)
    # plot.ModelSize_ParameterNumber_graph()
    # plot.ModelNumber_GPUusage_graph()
    # time.sleep(10)
    plot.Data_GPUusage_graph()