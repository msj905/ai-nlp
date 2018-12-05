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
from sklearn.naive_bayes import MultinomialNB


def naviebayes():


    news = fetch_20nresgroups(subset='all')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    tf = TfidfVectorizer()

    x_test = tf.reansform(x_test)


    x_train = tf.fit_transform(x_train)

    x_test = tf.fit_transform(x_test)


    mlt = MultinomialNB(alpha=1.0)

    print(x_train)

    mlt.fit(x_train,y_train)


    y_predict = mlt.predict(x_test)

    print("预测文章类别为: ",y_predict


    print("准确率: ",mlt.score()))


    return None



