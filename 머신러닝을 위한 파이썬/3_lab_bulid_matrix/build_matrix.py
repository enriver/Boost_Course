import numpy as np
import pandas as pd



'''
source는 row, target은 column 의 기준
source와  target의 정렬된 값을 활용하여 index 설정
rating 정보는 Matrix 에서 각 Element 값에 팔당
생성되는 Matrix Ndarray로 나타낼것
'''
def get_rating_matrix(filename, dtype=np.float32):
    path="C:\\Users\\river\\Desktop\\부스트코스\\머신러닝을 위한 파이썬\\3_lab_bulid_matrix\\"
    df=pd.read_csv(path+filename)
    
    return df.groupby(['source','target'])['rating'].sum().unstack().fillna(0).values


def get_frequent_matrix(filename, dtype=np.float32):
    path="C:\\Users\\river\\Desktop\\부스트코스\\머신러닝을 위한 파이썬\\3_lab_bulid_matrix\\"
    df=pd.read_csv(path+filename)
    
    return df.groupby(['source','target'])['target'].count().values.reshape(5,-5)

result_1=get_rating_matrix('movie_rating.csv')
result_2=get_frequent_matrix('1000i.csv')
