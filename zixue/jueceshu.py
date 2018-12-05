"""


if-then结构，利用这类结构分类学习方法

信息熵
划分依据是信息增益：当得知一个特征条件之后家少信息熵的大小


平方误差

基尼系数


api：class sklearn.tree.DecisionTreeClassifler(criterion='gini',max_depth=None,random_state=None)
criterion='gini', 选择信息增益熵
max_depth=None,  数的深度
random_state=None  随机种子

method   decision_path  返回决策树路径

结构保存
sklearn.tree.export_graphviz(estimator,out_file='tree.dot',feature_names=[","])


工具
apt install graphviz

命令
dot -Tpng tree.dot -o tree.png

案例：泰坦尼克号预测生死

流程

1。pd读取数据
2。选择有影响特征  处理缺失值
3。进行特征工程 pd转换字典 特征抽取

4。决策树估计器流程



数据准备时间少  缺点国语复杂


改进  剪枝（随机森林）
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def decision():


    titan = pd.read_csv("http://")


    x = [['plclass','age','sex']]
    y = ['survived']


    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


    dict = DictVrctorizer(sparse=False)

    x_train = dict.fit_transform(x_train.to_dict(orient="records"))

    print(disc.get_feture_names())

    x_test = dict.fit_transform(x_train_to_dict(orient="records")

#随机森林
    rf = RandomForestClassifier()

    param = {"n_estimators": [120.200,300,500,800,1200],"max_deph": [5,8,15,20,25,40]}



    #print(x_train)
    #决策树

    dec = DecisionTreeClassifier()
    dec.fit(x_train,y_train)

    print("准确率: ",dec.score(x_test,y_test)

    print("选择模型: ",gc.best_params_)

    export_graphviz(det,out_file="./",feture_names=['nianling','pclass=lst']))

    return None
if __name__ == "__main__":
    decision()










































