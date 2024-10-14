import pandas as pd

'''
    emotion 라벨화
'''

num_participant = 73
head_dir = 'data'
file_dir1 = 'c3_muse_stress_2021_total'
file_dir2 = 'c3_muse_stress_2022_total'
# file_dir3 = 'c4_muse_physio_2021_total'
# file_list = [file_dir1, file_dir2, file_dir3]
file_list = [file_dir1, file_dir2]


col = ['timestamp', 'segment_id', 'BPM', 'ECG', 'resp', 'arousal', 'valence']

thres = 0.05 # 임계값 0.1 -> 0.05 -> 0.01

'''
one='data/c3_muse_stress_2021_total/1.csv'
two='data/c3_muse_stress_2021_total/2.csv'
da1 = pd.read_csv(one)
da2 = pd.read_csv(two)
da2['deviceID'] = int(1)
print(one)
print(two)
total = pd.concat([da1, da2])
print(total)
total = total.dropna()
print(total)
'''



for file_dir in file_list:
    result = pd.DataFrame()
    for idx in range(1, num_participant+1):
        try:
            '''
                (방법1)
                arousal(+) & valence(+) : HVHA (excited)
                arousal(-) & valence(+) : HVLA (Calm)
                arousal(-) & valence(-) : LVLA (Sad)
                arousal(+) & valence(-) : LVHA (Upset) 
            '''

            file = 'data/' + file_dir + '/' + str(idx) + '.csv'
            data = pd.read_csv(file)
            emotion = []
            deviceID = []

            for idxx in range(len(data)):
                deviceID.append(idx)
                if  data.iloc[idxx, 5] > thres and data.iloc[idxx,6] > thres:
                    emotion.append('feliz') # HVHA (excited)
                elif data.iloc[idxx,5] < thres * (-1) and data.iloc[idxx,6] > thres:
                    emotion.append('calmado') # HVLA (Calm)
                elif data.iloc[idxx,5] < thres * (-1) and data.iloc[idxx,6] < thres * (-1):
                    emotion.append('triste') # LVLA (Sad)
                elif data.iloc[idxx,5] > thres and data.iloc[idxx,6] < thres * (-1):
                    emotion.append('enojado') # LVHA (Upset)
                else:
                    emotion.append('neutral')
            data['emotion'] = emotion
            data['deviceID'] = deviceID

            result = pd.concat([result,data])

        except:
            print(file_dir, '- There is no', idx, "participant's data")
    
    result = result.dropna()
    save_dir = 'data/' + file_dir + '_threshold.csv'
    result.to_csv(save_dir, index=None)

    print(result.describe())
