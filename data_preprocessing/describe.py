import pandas as pd
from collections import Counter

'''
    threshold로 재라벨화 한 emotion별 집계 보기
'''


# dir1 = 'data/c3_muse_stress_2021_total_threshold.csv'
# dir2 = 'data/c3_muse_stress_2022_total_threshold.csv'
dir1 = 'data/c3_muse_stress_2021_total.csv'
dir2 = 'data/c3_muse_stress_2022_total.csv'

data1 = pd.read_csv(dir1)
data2 = pd.read_csv(dir2)

print('2021:', Counter(data1['emotion']))
print('2022:', Counter(data2['emotion']))