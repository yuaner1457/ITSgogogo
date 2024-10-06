from sklearn.metrics import adjusted_rand_score

#接受的数据类型是两个list
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