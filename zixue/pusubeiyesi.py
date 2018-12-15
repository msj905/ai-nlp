""""
联合概率：多个条件同时成立

条件概率：A在另一个B已经发生条件下的发生概率



api：sklearn.naive_bayes.MultionmialNB(ALPHA = 1.0)


TITLE:
sklearn 20 news


1。加载  分割
2。 生成特征
3。 算法评估




"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from skleanr.metrics import classification_report


def naviebayes():


    news = fetch_20nresgroups(subset='all'
    #数据分割

    x_train, x_test, y_train, y_test = train_test_split(news.data,news.target, test_size=0.25)
    #对特征抽取
    tf = TfidfVectorizer()

    #x_test = tf.reansform(x_test)

    #以训练集中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)

    x_test = tf.fit_transform(x_test)

    #朴素贝叶斯算法
    mlt = MultinomialNB(alpha=1.0)

    print(x_train.toarray())

    mlt.fit(x_train,y_train)


    y_predict = mlt.predict(x_test)

    print("预测文章类别为: ",y_predict)


    print("准确率: ",mlt.score(x_test,y_test))

    print("每个类别的精确率和召回率: ", classification_report(y_test,y_predict,target_names=news.target_names)))
    return None



