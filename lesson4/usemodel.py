# encoding:utf8
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import gensim
from sklearn.manifold import TSNE



def tsne_plot():
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    print('开始为x、y赋值')
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    print('绘制散点图')
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


if __name__ == '__main__':

    model = gensim.models.Word2Vec.load('wiki.zh.text.model')
    word1 = u'数学'
    word2 = u'计算机'
    if word1 in model:
        print(u"'%s'的词向量为： " % word1)
        print(model[word1])
    else:
        print(u'单词不在字典中！')

    result = model.wv.most_similar(word2)
    print(u"\n与'%s'最相似的词为： " % word2)
    for e in result:
        print('%s: %f' % (e[0], e[1]))

    print(u"\n'%s'与'%s'的相似度为： " % (word1, word2))
    print(model.wv.similarity(word1, word2))

    print(u"\n'早餐 晚餐 午餐 中心'中的离群词为： ")
    print(model.wv.doesnt_match(u"早餐 晚餐 午餐 中心".split()))

    print(u"\n与'%s'最相似，而与'%s'最不相似的词为： " % (word1, word2))
    temp = (model.wv.most_similar(positive=[u'数学'], negative=[u'计算机'], topn=1))
    print('%s: %s' % (temp[0][0], temp[0][1]))

    tsne_plot()