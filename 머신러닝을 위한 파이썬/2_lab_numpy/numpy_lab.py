import numpy as np

''' n의 제곱수로 2 dimentional array를 생성하는 ndarray '''
def n_size_ndarray_creation(n, dtype=np.int):
    return np.array(range(n**2), dtype=dtype).reshape(n,n)

'''
shape이 지정된 크기의 ndarray를 생성
이때 행렬의 element는 type에 따라 0, 1 또는 empty로 생성됨
'''
def zero_or_one_or_empty_ndarray(shape, type=0, dtype=np.int):
    if type==0:
        return np.zeros(shape=shape, dtype=dtype)
    if type==1:
        return np.ones(shape=shape, dtype=dtype)
    if type==99:
        return np.empty(shape=shape, dtype=dtype)

'''
입력된 ndarray X를 n_row의 값을 row의 개수로 지정한 matrix를 반환함
이때 입력하는 X의 size는 2의 거듭제곱수로 전제함
만약 n_rows 가 1일 때는 matrix 가 아닌 vector로 반환함
'''
def change_shape_of_ndarray(X, n_row):
    if n_row==1:
        return X.flatten()
    else:
        return X.reshape(n_row,-1)

'''
입력된 ndarray X_1 과 X_2를 axis로 입력된 축을 기준으로 통합하여 반환하는 함수
X_1 과 X_2 는 matrix 또는 vector 임, 그러므로 vector 일 경우도 처리할 수 있어야함
axis를 기준으로 통합할 때, 통합이 불가능하면 False 가 반환됨
단 X_1 과 x_2 가 matrix,vector 형태로 들어왔다면 vector 를 row 가 1개인
matrix 로 변환하여 통합이 가능한지 확인할 것
'''
def concat_ndarray(X_1, X_2, axis):
    try:
        if X_1.ndim==1:
            X_1=X_1.reshape(1, -1)
        if X_2.ndim==1:
            X_2=X_2.reshape(1,-1)
        
        return np.concatenate((X_1, X_2), axis=axis)

    except ValueError as e:
        return False

'''
입력된 Matrix 또는 Vector를 ndarray X의 정규화된 값으로 변환하여 반환함
이때 정규화 변환 공식 Z = (X - X의평균) / X의 표준편차로 구성됨.
X의 평균과 표준편차는 axis를 기준으로 axis 별로 산출됨.
Matrix의 경우 axis가 0 또는 1일 경우, row 또는 column별로 Z value를 산출함.
axis가 99일 경우 전체 값에 대한 normalize 값을 구함.
'''
def normalize_ndarray(X, axis=99, dtype=np.float32):
    X=X.astype(np.float32)
    n_row, n_column= X.shape # 튜플 형태로 변환
    
    if axis==99:
        x_mean=np.mean(X)
        x_std=np.std(X)
        Z=(X-x_mean)/x_std
    
    if axis==0:
        x_mean=np.mean(X,0).reshape(1,-1)
        x_std=np.std(X,0).reshape(1,-1)
        Z=(X-x_mean)/x_std

    if axis==1:
        x_mean=np.mean(X,1).reshape(n_row,-1)
        x_std=np.std(X,1).reshape(n_row,-1)
        Z=(X-x_mean)/x_std

    return Z

'''입력된 ndarray X를 argument filename으로 저장함'''
def save_ndarray(X, filename="test.npy"):
    file_test=open(filename, 'wb')
    np.save(X, file_test)

'''
입력된 ndarray X를 String type의 condition 정보를 바탕으로
해당 컨디션에 해당하는 ndarray X의 index 번호를 반환함
단 이때, str type의 조건인 condition을 코드로 변환하기 위해서는
eval(str("X") + condition)를 사용할 수 있음
'''
def boolean_index(X, condition):
    condition=eval(str('X')+condition) # eval : 입력받은 값을 숫자형태로 변환해줌(산술 연산 가능)
    return np.where(condition)

'''
입력된 vector type의 ndarray X에서 target_value와 가장 차이가 작게나는 element를 찾아 리턴함
이때 X를 list로 변경하여 처리하는 것은 실패로 간주함.
'''
def find_nearest_value(X, target_value):
    return X[np.argmin(np.abs(X-target_value))]

'''입력된 vector type의 ndarray X에서 큰 숫자 순서대로 n개의 값을 반환함.'''
def get_n_largest_values(X, n):
    return X[np.argsort(X[::-1])[:n]] # arg 붙어있는 메소드 -> index 반환