from sklearn.metrics import normalized_mutual_info_score

#这个函数也是接受list，其中x是咱自己的模型聚类出来的label，y是人家测试集的label
def NMI(x,y):
    if isinstance(x,list) and isinstance(y,list):
        return normalized_mutual_info_score(x,y)
    else:
        print(f'NMI:Wrong type:need list')
        return 2

if __name__ == '__main__':
    x=[1,2,3,4,5]
    y=[1,2,2,4,5]
    print(NMI(x,y))