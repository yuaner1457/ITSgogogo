import pandas as pd
import numpy as np

def trans(row_data):
    row=row_data['coordinate']
    row=row.replace('[','').replace(']','').replace('\n',' ')
    array=np.fromstring(row,sep=' ')
    array=array.reshape(array.size//3,3)
    array=array.astype(float)
    row_data['coordinate']=array
    return


#这个函数用于将相应.csv文件中的字符串类型的数据更改为numpy形式，字符串形式太恶心了
def trans_data(filename):#filename就是需要读入的文件的名字
    data=pd.read_csv(filename)
    data=data.apply(trans, axis=1)
    return data

if __name__=='__main__':
    filename="washed_data_50_1.csv"
    data=trans_data(filename)
    print(data)