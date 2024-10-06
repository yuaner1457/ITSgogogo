from sklearn.metrics import adjusted_rand_score

#接受的数据类型是两个list，其中x是咱们的模型聚类出来的标签，y是实际的label，输出在（0,1），越靠近1越好
def ARI(x,y):
    if isinstance(x,list) and isinstance(y,list):
        return adjusted_rand_score(x,y)
    else:
        print(f'Wrong type:need list')
        return 2

if __name__ == '__main__':
    x=[0,0,1,1,2,2]
    y=[0,0,1,2,2,2]
    print(ARI(x,y))