# 机场终端区航空器进离场航迹聚类分析

## 背景介绍

> 机场终端区是设立在一个或几个邻近繁忙机场上空的空中交通管制区域，是实现航空器进场、离场安全高效有序运行的关键区域，也是飞行密度最大、空域环境最复杂、管制指挥难度最大的空中交通服务空域。繁忙机场终端区空域进离场航线密集交织、潜在飞行冲突风险，航空器起降飞行易受复杂气象等空域环境影响，导致航空器难以按照指定的标准进离场飞行程序飞行，进离场飞行航迹偏差时有发生。为提高机场终端区运行效率和安全水平，空管系统须实时监视并准确识别终端区内每一架航空器飞行航迹偏差和真实飞行意图。针对这一需求，国内外学术界和产业界正积极引入大数据、机器学习等新技术，开展机场终端区航空器进离场航迹聚类分析研究。
>
> 在此背景下，由于缺乏统一量化评测标准和性能比较基准，相关研究的深入开展和成果应用受到了一定程度的制约。本赛题旨在通过提供统一的机场终端区航迹仿真数据集和人工标注的聚类标签，构建进离场航迹聚类量化评价关键指标，鼓励各参赛团队面向实际应用需求，创新设计航迹聚类方法，通过量化评估和性能比较，汇聚行业内外创新资源，共同完善进离场航迹聚类分析的评测标准，推动相关技术实际应用落地。

## 问题分析

首先，我们组分析了机场终端区的航线相比于空域中航线的不同之处。在[*A Spatial-Temporal Neural Network Model for Estimated Time of Arrival Prediction of Flights in Terminal Manoeuvring Area*](.\Ma 等 - A Spatial-Temporal Neural Network Model for Estimated Time of Arrival Prediction of Flights in Termi.pdf)这篇文章中，作者详尽地介绍了目前机场终端区航线聚类分析这一研究方向的进展。由于作者描述的已经尽善尽美，我们决定将介绍的原文复制：

> Based on these studies, models that combine clustering with machine learning predictors have been applied to ETA prediction in recent years to further improve accuracy. Hong and Lee used dynamic time warping (DTW) to identify major trajectory patterns and applied multiple linear regression (MLR) for each pattern topredict the ETA of flights from a specified entry point [35]. Wang et al. clustered the trajectories into several patterns by the density-based spatial clustering of applications with noise (DBSCAN) and then trained an individual neural network (NN)-based model for each cluster [33].

通过阅读领域相关的大量文献，我们认为，本项目所面临的主要问题如下：

#### 1.距离函数

在[*Trajectory Prediction for Vectored Area Navigation Arrivals*](.\Hong和Lee - 2015 - Trajectory Prediction for Vectored Area Navigation Arrivals.pdf)这篇文章中，作者论证了在使用三维坐标表示航迹点，通过一组不同时间的航迹点来描述航线，通过计算两条航迹对应点的欧式距离并将其作为最终损失进行聚类的这种方法在机场终端区是没有可操作性的。主要有两方面原因。第一，如果认为两条航线的距离是相同时间间隔对应点的欧氏距离的累加，那么由于天气，调度等原因造成的时间序列的微小误差，会因为大量的点的累计而放大到足以影响聚类结果的程度。刚才提到文章中的这一段提到了这一问题：

> Since the primary purpose of trajectory clustering in this paper is to identify the vectoring patterns of air traffic controllers, the spatial shape ofthe trajectory is more important than the speed or acceleration of the aircraft along the trajectory. Note that, in the proposed method, the speed variation of the aircraft movements along the trajectory can be considered by the regression models of travel time, which will be discussed laterin the paper. The previously suggested clustering approaches are inappropriate for the proposed method because, in those approaches, a small misalignment in time would result in a large distance between the trajectories. As an example, two sets of trajectory pairs are shown in Fig. 6. Inthis figure, each trajectory is represented by a time sequence of aircraft positions, and the lengths of all sequences are identical. Although thetrajectories in Fig. 6a are more similar in shape to each other than the trajectories in Fig. 6b, the simple application of the Euclidian distancebetween the sequences results in less similarity (i.e., a larger distance) between the trajectories in Fig. 6a than between those in Fig. 6b. In fact, asmall misalignment in the early parts of the two trajectories in Fig. 6a introduces the continuing discrepancy in subsequent points of the sequences.

针对这一问题，我们迫切地需要一种可以有效处理时域离散点的距离函数，很幸运，我们找到了这么一种距离计算方法：[DTW动态时间规整](.\DTW.pdf)。该算法通过动态规划的方式，可以找到两条航线之间的最短的一一对应方式。我们通过python语言实现了这个方法，代码如下：

```python
import numpy as np

def loss(x1,x2):
    final=np.abs(x1-x2)
    final=np.sum(final,axis=1)#应该按照行进行相加
    return final

def Findpace(D):
    list=[]
    answer=0
    end=D.shape#找到终点的坐标(在迭代过程中会当做目前终点使用)
    x=end[0]
    y=end[1]
    end=(x-1,y-1)
    answer=D[end]
    list.append(end)
    while end != (0,0):
        x=end[0]
        y=end[1]
        D[end]=np.inf
        tem_idx=np.argmin(D[x-1:x+1,y-1:y+1])
        end=(int(x-1+tem_idx/2),y-1+np.mod(tem_idx,2))
        list.append(end)
    return answer,list

#这个函数接受两个输入，sample，std两条时间路径序列，输出是一个浮点数，即其dtw偏差；请注意，两个输入要求为numpy.array
def DTW(sample,std):
    l1=sample.shape[0]
    l2=std.shape[0]
    D=np.full((l1+1,l2+1),np.inf)#D是损失矩阵
    D[0,0]=0
    for i in range(1,l1+1):#进入loss中计算的时候需要减一
        C=loss(sample[i-1],std)
        for j in range(1,l2+1):
            D[i,j]=np.min(D[i-1:i+1,j-1:j+1])+C[j-1]
    lo,path=Findpace(D)
    return lo#,path

if __name__ == '__main__':
    sample=np.array([[0,0],[0,2],[2,1]])
    std=np.array([[0,0],[0,1],[0,2],[2,1]])
    lo,path=DTW(sample,std)
    print(f"最终的损失是:{lo},最终的路径是:{path}")
```

这段代码不仅可以返回最终的距离，还可以返回两段航迹之间的最佳对应方式。我们使用两条航线进行了测试，测试效果如下：

![DTW实际效果](.\firstbaby.png)

可以看到，DTW完美地实现了我们想要找到最短路径点的要求。最终我们在原有DTW算法上做出改进，优化了时间复杂度并作为最终的距离判断函数。

但是即便如此，算法的时间复杂度还是太高，最终我们通过提取特征并将航迹展平为高维向量的方式，将航迹简化为高维特征向量，在几乎不损失外部指标的情况下，极大提升了时间效率。具体实现在**伪代码**部分给出。

#### 2.聚类模型

聚类模型的选择又是一个不小的挑战。在[*Algorithms for hierarchical clustering: an overview*](.\Murtagh和Contreras - 2012 - Algorithms for hierarchical clustering an overview.pdf) 这篇文章中，作者介绍了目前的主流聚类模型，并且详细介绍了层次聚类模型的分类和发展。

###### a.中心聚类模型

中心聚类方法通过确定每个簇的中心（质心）来进行数据分组，常见的算法有 K-Means 和 K-Medoids。

**K-Means**

> 步骤：
> 选择 K 个初始中心点（质心）。
> 将每个数据点分配给距离其最近的质心。
> 更新质心位置为分配到该簇的所有点的平均值。
> 重复步骤 2 和 3，直到质心不再改变或达到最大迭代次数。
> 优点： 计算速度快，适用于大规模数据。
> 缺点： 需要预先指定 K 值，容易受到初始质心选择的影响，对噪声和离群点敏感。

**K-Medoids**

> 步骤： 类似于 K-Means，但质心被替换为簇内的实际点（Medoid），更具鲁棒性。
> 优点： 对离群点更不敏感。
> 缺点： 计算复杂度较高。

中心聚类模型的聚类结果有可能受随机初始点的初始位置的影响，存在不稳定的风险。并且在本题中，使用中心聚类方法由于数据并非单个维度上的点，其聚类效果并不好，因此在本项目中我们没有采用这个方法

###### b.密度聚类模型

密度聚类方法根据数据点的密度来定义簇，常见算法有 DBSCAN 和 OPTICS。

**DBSCAN（Density-Based Spatial Clustering of Applications with Noise）**

> 步骤：
> 定义邻域的半径 ε 和最小点数 MinPts。
> 对于每个未访问的点，检查其邻域内的点数。
> 如果邻域内的点数大于或等于 MinPts，将这些点标记为同一簇。
> 重复步骤 2 和 3，直到所有点被访问。
> 优点： 能够发现任意形状的簇，能够处理噪声点。
> 缺点： 参数选择（ε 和 MinPts）敏感，对数据分布有一定假设。

**OPTICS（Ordering Points To Identify the Clustering Structure）**

> 步骤： 类似于 DBSCAN，但生成一个可扩展的簇结构，提供不同密度的聚类。
> 优点： 适用于不同密度的簇，更灵活。
> 缺点： 计算复杂度较高。

密度聚类模型解决了中心聚类模型聚类结果不稳定的问题，但是在我们实际的尝试中，相关参数的训练和选取太过困难，且目前没有明确的度量方法找到最佳的参数，因此最终在进行了大量的尝试后，我们最终放弃了这个方案

###### c.层次聚类模型

层次聚类（Hierarchical Clustering）是一种无监督学习的聚类技术，通过构建数据的层次结构来进行聚类分析。它可以生成不同层次的聚类结果，通常以树状图（Dendrogram）的形式展现。这种方法适用于需要理解数据内部结构和层次关系的场景。下面将详细介绍层次聚类的基本概念、类型、算法步骤、优缺点及应用场景。

**凝聚型（Agglomerative）**

> 步骤：
> 将每个点视为一个独立的簇。
> 找到距离最近的两个簇并合并，形成新的簇。
> 重复步骤 2，直到所有点都合并成一个簇或达到设定的簇数。
> 优点： 不需要预先指定簇的数量，能够展示数据的层次结构。
> 缺点： 计算复杂度高（O(n²)），对噪声敏感

**分裂型（Divisive）**

> 步骤： 从一个大簇开始，逐步将其分裂为更小的簇。
> 优点： 可以捕捉更大的全局结构。
> 缺点： 通常不如凝聚型常用，计算复杂度更高。

这两种方式也被称为**自上而下型**和**自下而上型**。

这种聚类模型聚类结果稳定，效率高且最终测试效果好。因此我们最终选择层次聚类模型作为我们的聚类方式。

#### 3.数据处理

本项目的另一大难点在于数据量庞大，运行时间长。为了降低数据量，我们根据[*Schematizing Maps: Simplification of Geographic Shape by Discrete Curve Evolution*](.\Barkowsky 等 - 2000 - Schematizing Maps Simplification of Geographic Shape by Discrete Curve Evolution.pdf)这本书中提出的一种提取航线特征的方法，对航线进行了特征提取，代码如下：

```python
import pandas as pd
from rdp import rdp
import Baby_washdata_change as wdc
import numpy as np


def calculate_deviation(original_path, simplified_path):
    original_path=original_path.loc['coordinate']
    simplified_path=simplified_path.loc['coordinate']
    deviations = 0
    for point in original_path:
        # 计算每个点到简化路径的最小距离
        min_distance = np.min(np.sqrt(np.sum((simplified_path - point) ** 2, axis=1)))
        deviations=deviations+min_distance
    # 返回平均偏离度
    return deviations

def simplify(x,epsilon):
    data=x['coordinate']
    data_simple=rdp(data,epsilon)
    x['coordinate']=data_simple
    return x

if __name__ == '__main__':
    data_init=wdc.trans_data("washed_data_50_1.csv")
    see=[]
    size=[]
    for i in np.arange(0.03,0,-0.0001):
        epsilon=i
        data=data_init.apply(lambda row: simplify(row,epsilon),axis=1)
        asize=data.apply(lambda row: len(row['coordinate']),axis=1)
        asize=np.mean(asize)
        size.append(asize)
        loss_num=data_init.apply(lambda row: calculate_deviation(row,data.loc[row.name]),axis=1)
        loss_num=np.mean(loss_num)
        see.append(loss_num)
        print(f'正在计算{i}/1->0')
    see=pd.DataFrame(see,columns=['See'])
    size=pd.DataFrame(size,columns=['size'])
    see.to_excel('see.xlsx')
    size.to_excel('size.xlsx')
```

通过这种方式，我们不仅可以对航迹进行简化，更能够通过训练的方法找到**简化航迹**和**保持航迹特征**之间的平衡点。最终我们选择0.013作为航迹简化阈值。通过这种方式，我们可以用原来10%的数据量来概括航迹95%的特征。该方法可以有效降低计算量。

不仅如此，我们尝试了多种可以加快计算速度的方法，包括但不限于并行计算，向量化的方式，最终成功将计算时间降低至原方案的16%，极大优化了时间复杂度。我们同时通过预处理将数据存入表格的方式来降低训练过程中消耗的时间。

## 伪代码

```
开始

定义函数 resample_flight_path(df, num_points):
    如果 df 包含 't', 'x', 'y', 'z' 列:
        获取 t_vals, x_vals, y_vals, z_vals
        创建插值函数 interp_x, interp_y, interp_z
        创建新的时间点 t_new（均匀分布 num_points 个点）

        resampled_path = 连接插值后的 x, y, z 值
        返回 resampled_path
    否则:
        抛出错误 "输入数据缺少必要的列：'t', 'x', 'y', 'z'"

尝试:
    设置文件路径为 'valid_data.csv'
    读取数据到 data

    设置 num_points 为 230
    对于每个 flight_id，使用 resample_flight_path 进行重采样
    将结果存储为 flight_features_resampled

    将重采样的特征矩阵转换为 numpy 数组 flight_features_matrix_resampled

    # 标准化特征
    创建 StandardScaler 实例
    对 flight_features_matrix_resampled 进行标准化，得到 flight_features_scaled_resampled

    # 使用 PCA 进行降维，保留 95% 的方差
    创建 PCA 实例，设置 n_components 为 0.95
    对 flight_features_scaled_resampled 进行 PCA 变换，得到 flight_features_pca

    打开输出文件 '输出说明.txt' 以写入模式:
        # 写入 PCA 后的相关信息

        # 层次聚类
        进行层次聚类，使用 linkage 方法（Ward 方法）
        设置最大距离 max_d 为 100
        使用 fcluster 生成聚类标签 flight_clusters_hierarchical

        创建 flight_features_hierarchical_df 数据框，包含 flight_id 和 cluster

        计算簇的数量 num_clusters
        输出簇的数量

        尝试:
            计算轮廓系数 silhouette_avg
            输出轮廓系数
        捕获错误:
            输出错误信息

        尝试:
            计算戴维斯-鲍尔丁指数 dbi_score
            输出戴维斯-鲍尔丁指数
        捕获错误:
            输出错误信息

    # 将聚类结果输出到文件
    设置输出文件名为 'clustered_flight_results_pca.csv'
    将 flight_features_hierarchical_df 保存为 CSV 文件
    输出文件保存成功信息

    打开输出文件 '输出说明.txt' 以追加模式:
        写入聚类结果文件保存信息

捕获所有异常:
    输出处理过程中发生的错误
    将错误信息写入 '输出说明.txt'

结束
```

我们最终的输出分为两部分：txt文件包含内部指标的结果，csv文件里面包含flight_id和聚类结果。

## 评价和测试

#### 1.评价指标

**内部指标**：我们通过轮廓系数（SC）、戴维斯-鲍尔丁指数（DBI）这两个判定方法对训练集上的数据的聚类效果进行测试。

**外部指标**：通过基于验证集标签的调整兰德指数（ARI）和归一化互信息（NMI）对模型在验证集上的效果进行测试。

#### 2.测试结果

```
PCA降维后保留的主成分数量:6
共有 16 个轮廓系数(silhouette Score):0.6328015114151609
戴维斯-鲍尔丁指数(Davies-Bouldin Index):0.49195058406489023
合并后的数据集大小: 1690
聚类特征矩阵的大小: 1690
NMI:0.9241224226601908
ARI: 0.8771178737071348
聚类结果已保存至:clustered_flight_results_pca.csv
```

上面的结果是我们通过实际的模型在验证集上得出的。

#### 3.代码和数据

在这次的项目中，我们小组成员首次使用git的方式进行合作。在合作过程中，我们所有的数据都保存在[这里](https://github.com/yuaner1457/ITSgogogo)我们的实际工作链如下图所示：

![](.\微信图片_20241007194332.png)

通过git进行合作可以对我们进行高效的协同有很大的帮助

## 潜在应用分析

1. **航空数据分析**
  航迹模式识别：航空公司可以利用该代码分析不同航班的飞行轨迹，识别不同航班的飞行模式，以优化航班安排和提高安全性。
  飞行路径优化：通过聚类分析，航空公司可以识别出相似的飞行路径，从而优化航线规划，减少燃料消耗和航班延误。
2. **交通管理**
  交通流量分析：城市交通管理部门可以利用类似的技术分析交通流量，识别拥堵区域，并制定改善措施。
  智能交通系统：通过对实时交通数据进行分析，改进智能交通信号控制，提升道路使用效率。
3. **数据科学与机器学习**
  特征工程：在机器学习模型训练之前，特征标准化和降维是非常重要的步骤，可以提高模型性能。该代码为数据预处理提供了良好的模板。
  聚类分析：可应用于用户行为分析、市场细分等领域，帮助企业识别客户群体和个性化服务。
4. **农业与环境监测**
  农田监测：通过重采样与分析农业无人机收集的飞行数据，可以监测农田健康、作物生长状态等。
  环境监测：分析无人机或卫星收集的环境数据，帮助监测生态变化、气候变化等问题。
5. **机器人与自动驾驶**
  导航与路径规划：在机器人和自动驾驶车辆中，轨迹分析和重采样可以帮助改进路径规划算法，提高导航精度和效率。
  碰撞检测与避障：通过对周围环境的轨迹数据分析，机器人可以预测障碍物运动，进行实时避障。
6. **安全与安防**
  行为模式识别：在安防领域，分析航迹数据可以识别可疑行为，如异常航迹或不符合常规的飞行模式，提升安全监控能力。
  飞行安全监控：可用于监测无人机等飞行器的飞行行为，确保飞行安全。
7. **科学研究**
  气象研究：在气象学中，飞行数据可以帮助研究天气模式、气流及其对飞行的影响。
  地理信息系统（GIS）：为地理空间数据分析提供支持，有助于城市规划、环境监测等研究。
8. **预测模型的构建**
  基于数据的预测：结合聚类分析结果，可以构建针对特定群体或行为的预测模型，提高决策的准确性。

