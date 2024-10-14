import pandas as pd

'''
    따로따로 나뉘어져 있는 파일들 하나로 합치기
'''

num_participant = 73
file_dir1 = 'c3_muse_stress_2021'
file_dir2 = 'c3_muse_stress_2022'
file_dir3 = 'c4_muse_physio_2021'
# file_dir4 = 'raw-data-ulm-tsst_2021'
file_list = [file_dir1, file_dir2, file_dir3]

sub_dir1 = 'feature_segments'
sub_dir2 = 'label_segments'
# sub_dir3 = 'metadata'

col = ['timestamp', 'segment_id', 'BPM', 'ECG', 'resp', 'arousal', 'valence']


for file_dir in file_list:
    for idx in range(1, num_participant+1):
        try:
            total_file_name = file_dir + '/' + file_dir + '_total/' + str(idx) + '.csv'

            name_bpm = file_dir + '/' + sub_dir1 + '/BPM/' + str(idx) + '.csv' 
            name_ecg = file_dir + '/' + sub_dir1 + '/ECG/' + str(idx) + '.csv' 
            name_resp = file_dir + '/' + sub_dir1 + '/resp/' + str(idx) + '.csv' 
            if file_dir == file_dir1:
                name_arousal = file_dir + '/' + sub_dir2 + '/arousal/' + str(idx) + '.csv' 
                name_valence = file_dir + '/' + sub_dir2 + '/valence/' + str(idx) + '.csv'
                on_col = ['timestamp', 'segment_id'] 
            elif file_dir == file_dir2:
                name_arousal = file_dir + '/' + sub_dir2 + '/physio-arousal/' + str(idx) + '.csv' 
                name_valence = file_dir + '/' + sub_dir2 + '/valence/' + str(idx) + '.csv'        
                on_col = ['timestamp', 'subject_id']      
            else:
                name_arousal = file_dir + '/' + sub_dir2 + '/anno12_EDA/' + str(idx) + '.csv'
                on_col = ['timestamp', 'segment_id'] 
            
            bpm = pd.read_csv(name_bpm)
            ecg = pd.read_csv(name_ecg)
            file1 = pd.merge(bpm, ecg, how='outer', on=on_col)
            resp = pd.read_csv(name_resp)
            file2 = pd.merge(file1, resp, how='outer', on=on_col)
            arousal = pd.read_csv(name_arousal)
            file3 = pd.merge(file2, arousal, how='outer', on=on_col)

            if file_dir == file_dir1 or file_dir == file_dir2:
                valence = pd.read_csv(name_valence)
                file = pd.merge(file3, valence, how='outer', on=on_col)
                file.columns = col
            
            if file_dir == file_dir1 or file_dir == file_dir2:
                print(file_dir, idx, "participant's data is merged")
                file.to_csv(total_file_name, index=None)
            else:
                print(file_dir, idx, "participant's data is merged")
                file3.to_csv(total_file_name, index=None)
        except:
            print(file_dir, '- There is no', idx, "participant's data")