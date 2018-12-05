
"""
数据集划分
训练数据：用于训练构建模型  75%
测试数据：用于模型校验使用 评估模型是否有效   25%

api：sklearn.model_selection.tain_test_split

sklearn.datasets  加载获取流行数据集
datasets.load_*()  获取小规模数据集

datasets.fetch_*(data_home=None)  获取大规模数据集，需从网上下载 ata_home数据集下载目录  默认在家目录

load_*  fetch_*  返回数据类型datasets.base.Bunch字典格式

data:特征数据组  二维数组  numpy.ndarray
target:标签组名
DESCR:数据描述
feature_names:特征名字
target_names:标签名


fenge:

sklearn.model_selection.tain_test_split(*arrays,*options)

x  特征值
y   标签值

test_size  测试集大小

random_state  随即种子

return  训练集特征值 测试集特征值 训练标签测试集标签



大数据集：
skleanr.datasets.fetch_20newsgroups(data_home=None,subset='train')

train   训练

test    测试

all    全部


datasets.clear_data_home(ata_home=None) 清楚数据




特征工程

实例化transformer

调用fit_tansform


estmator估算其

api:

分类：
sklearn.neighbors  近邻
sklearn.naive_bayes   贝叶斯
sklearn.linear_model.LogisticRegression  逻辑回归
sklearn.tree    决策树随即森林

回归

sklearn.liner_model.LineaRegression  线性回归
sklearn.model.Ridge  岭回归

离散性数据  区间不可分
连续性数据  区间可分



监督


分类： 近邻  贝叶斯  决策树  随机森林 逻辑回归  神经网络
回归  线性回归  岭回归
标注  隐马尔可夫模型

非监督
类聚  k-means




k近邻算法（knn）定义：如果一个样本在特征空间中的k个最相似得 样本宗大多数属于某一个类别，则该样本也属于这个类别，需要做标准化
公式(欧式距离)： (a1-b1)^2+(a2-b2)^2+(a3-b3)^2)

相似的样本，特征之间的值也相似

api:sklearn.neighbors.KNneighborsClassifier(n_neighbors=5,algorithm='auto')
n_neighbors=5  查询默认使用的邻居数  algorithm='auto'用于计算最近邻居的算法 balltree使用Ballree  kdtree使用KDTree  auto 尝试传递fit方法的值来决定最适合算法


"""
"""
题目：www.kaggle.com/c/facebook-v-predicting-check-ins
流程：

特征值：x，y纵坐标定位准确性  目标值入驻位置
处理：时间处理
    1。
1。缩小数据范围
DataFrame.query()
2。处理日期数据
pd.to_datetime
pd.Datetimeindex
3。增加分割的日期数据
4。删除没用的数据
5。将签到位置少于n的删除
place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)



算法问题：
k去很小，容易一场殿影响
k取之大，k影响

优点：简单 易于理解 易于实现 无需计算参数 
缺点：优懒惰算法，对测试样本分类计算量大，内存开销大 必须制定k值 k值选择不当精度不能保证  使用小数据场景，几千到几万样本


分类评估模型
准确率 预测结果的正确的百分比estmtor。score（）
精确率：预测结果为正例样本中真实的比例（查的准）
召回率：真实为正例样本中预测为正例的比例（查的全，对正样本的区分能力）
混淆矩阵

其他分类标准 f1-score  反映了模型的稳健性

 api：sklearn.metrics.classification_report(y_true,y_pred,taget_names)
 
 y_true,真实目标值
 y_pred,估计器预测目标值
 taget_names 目标类别名称
 return  每个类别精确吕和召回率


模型选择与调优

交叉验证：为了让被评估的模型更加准确，一般选择10次的交叉验证


网格搜索（超参数搜索）：通常情况下，有很多需要手动指定（如k值）这种超参数。但手动过程繁杂，所以需要对模型几种超参数组合，每组超参数都采用交叉验证来进行评估，最后选出最优组合建立模型

api：skearn.model_selection.GridSearchCV

"""

from sklearn.datasets import load_iris,fetch_20newsgroups,load_bostn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from skleanr.metrics import classification_report
import pandas as pd

li = load_iris()

def knncls():
    """
    k-jinlin签到位置
    :return: None
    """
    #读取数据
    data = pd.read_csv("./a.csv")

    print(data.head())
    # 处理数据数据
    #缩小数据，查询数据
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75")
    #time
    time_value = pd.to_datetime(data['time'],unit='s')

    print(time_value)
    #zhuanhuan disc
    time_value = pd.DatetimeIndex(time_value)
#tezheng
    data['day'] = time_value.day
    data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    data.drop(['time'],axis=1)

#shaoyu n del
    place_count = data.groupby('place_id').count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)

    #tezheng   mubiao

    y = data['place_id']

    x = data.drop(['place_id'],axis=1)

    #fenge
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

    #标准化
    x_tarin = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)

    #tezheng gongcheng

    knn = KNeighborsClassifier(n_neighbors=5)
    #fit
    knn.fit(x_train,y_train

    #canshuzhi
    param = {"n_neighbors": [3,5,10]}

    #wangge
    gc = GridSearchCV(knn,param_grid=param,cv=2)
    gc.fit(x_train,y_train)

    #yuce jieguo

    y_predict = knn.predict(x_test)
    print("预测目标其拿到位置： "y_predict)
    #
    print("预测准确率: ",knn.score(x_test,y_test)
    print("交叉验证最好结果: ", gc.best_score_)
    print("选择最好模型: ", gc.best_estimator_)
    print("每个超参数每次交叉验证的结果: ", gc.cv_results_)

if __name == "__main__":
    knncls()



