import pandas as pd
import numpy as np

'''
    hrdata : [time, hr, imei]
    emotion : [eTimeStamp, emotion, activity, deviceID]

    hr data & emption data 각각 나누기
'''

dir1 = 'data/c3_muse_stress_2021_total'
dir2 = 'data/c3_muse_stress_2022_total'
dir_list = [dir1, dir2]

for dir_name in dir_list:
    data = pd.read_csv(dir_name+'.csv')

    # hr data
    hrdata = data.loc[:, ['timestamp', 'BPM', 'deviceID']]
    hrdata.columns = ['time', 'hr', 'imei']
    hrdata.to_csv(dir_name+'_hr.csv', index=None)

    # emotion
    emotion = data.loc[:, ['timestamp', 'emotion', 'deviceID']]
    emotion.columns = ['eTimeStamp', 'emotion', 'deviceID']
    emotion['activity'] = 'no'
    emotion.to_csv(dir_name+'_emotion.csv', index=None)



    
