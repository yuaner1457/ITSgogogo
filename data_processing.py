import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def data_processing(a=0.1):#a为筛选系数
    t_step = 5

    def find_exception(df):
        exceptional_id = []
        for flight_id, group in df.groupby('flight_id'):
            t = np.array(group['t'])
            # 生成等差数列
            arithmetic_sequence = np.arange(t[0], t[0] + t_step * len(t), t_step)
            # 计算误差
            errors = t - arithmetic_sequence
            # 计算绝对误差
            mae = np.mean(np.abs(errors))
            if mae / len(t) > a:
                exceptional_id.append(flight_id)
        return exceptional_id

    df = pd.read_csv('train_data.csv')
    df['t'] = df['t'].astype(float)
    exception_id = find_exception(df)
    # 清除异常航线
    df = df[~df['flight_id'].isin(exception_id)]
    # 将 x, y, z 列整合成一个向量
    df['coordinate'] = df.apply(lambda row: np.array([row['x'], row['y'], row['z']]), axis=1)
    # 删除原始的 t ,x, y, z 列
    df.drop(columns=['t', 'x', 'y', 'z'], inplace=True)
    df.dropna(inplace=True)
    df = df.groupby('flight_id')['coordinate'].apply(lambda x: np.vstack(x)).reset_index()
    df.set_index('flight_id', inplace=True)
    # 进行Z-score 标准化
    scaler = StandardScaler()
    df['coordinate'] = df['coordinate'].apply(lambda x: scaler.fit_transform(x))
    return df
if __name__ == '__main__':
    print(data_processing(0.1).head())