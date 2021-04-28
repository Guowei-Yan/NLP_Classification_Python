import numpy as np
!pip --version
import pandas as pd
import os
import regex as re
'''for dirname, _, filenames in os.walk("D:\Linux\py\machine_learning\kaggle"):
    for filename in filenames:
        print(os.path.join(dirname, filename))'''
import multiprocessing
multiprocessing.set_start_method('forkserver')

train_data = pd.read_csv("comments_train.csv")
train_data.head(5)
label = train_data["label"]
features = train_data["comment"].copy()

from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action = 'ignore')

import gensim
from gensim.models import Word2Vec

import jieba
from ckip import CkipSegmenter
list(bb)[1]
test_data = pd.read_csv("comments_test.csv")
with open("kaggle.txt","w",encoding="utf-8") as ff:
    for i in range(len(features)):
        rule=re.compile(r'[ 0-9a-zA-z\P\W]')
        features[i] = np.array(" ".join(jieba.cut(rule.sub('',features[i]))).split())
        ff.write(' '.join(features[i])+"\n")
    for j in range(len(test_data)):
            rule=re.compile(r'[ 0-9a-zA-z\P\W]')
            strings = np.array(" ".join(jieba.cut(rule.sub('',test_data.loc[j,"comment"]))).split())
            ff.write(' '.join(strings)+"\n")
    ff.close()

'''for i in range(len(features)):
    rule=re.compile(r'[ 0-9a-zA-z\P\W]')
    features[i] = np.array(" ".join(jieba.cut(rule.sub('',features[i]))))'''
'''    for j in range(len(test_data)):
        rule=re.compile(r'[ 0-9a-zA-z\P\W]')
        strings = np.array(" ".join(jieba.cut(rule.sub('',test_data.loc[j,"comment"]))).split())
        ff.write(' '.join(strings)+"\n")'''
features = np.array(features)
features[0][0]



from gensim.models import word2vec
from gensim import models

bb = word2vec.Text8Corpus("kaggle.txt")


model = Word2Vec(size=400, window=12, min_count=1, workers=11,)
model.build_vocab(bb)
model.intersect_word2vec_format('word2vec_779845.bin',binary=True)
model.train(bb,total_examples=model.corpus_count,epochs=150)

'''model.vocabulary.sample
model.vocabulary.raw_vocab'''

model.wv.save_word2vec_format("model.bin", binary=True)
model.wv["很卡"]
model.most_similar("很卡")


featuress = features.copy()
for i in range(len(featuress)):
    lst = []
    aa = np.array(list(map(lambda x : model.wv[x],featuress[i])))
    lst = [0]*len(aa[0])
    for j in range(len(aa)):
        lst += aa[j]
    featuress[i] = lst/len(aa)

len(set(label))
set(label)

import pandas as pd
df = pd.DataFrame(index = range(len(featuress)),columns=range(len(featuress[0])))
for i in range(len(featuress)):
    df.iloc[i,:] = featuress[i]

from sklearn.decomposition import PCA

pca = PCA(n_components=4)
p_features = pca.fit_transform(df)
df2 = pd.DataFrame(p_features)
df2

from sklearn.ensemble import RandomForestClassifier
# 这里构建 100 棵决策树，采用信息熵来寻找最优划分特征。
rf_model = RandomForestClassifier(
    n_estimators=100, max_features=None, criterion='entropy',verbose=1,n_jobs=-1)

rf_model.fit(df2,label)

import sklearn.model_selection as ms
import sklearn.ensemble as ensemble
import sklearn as sk
from sklearn.neural_network import MLPClassifier
import sklearn.neighbors as knn

model4 = knn.KNeighborsClassifier(n_neighbors=100,n_jobs=-1)
model2 = ensemble.AdaBoostClassifier(n_estimators=100)
model3 = sk.svm.SVC(kernel="poly", )
model5 = MLPClassifier(hidden_layer_sizes=(200,100,50),activation="rbf",solver="lbfgs")
ms.cross_val_score(model3, df, label, groups=None, scoring=None, cv=5, n_jobs=-1, verbose=1, fit_params=None).mean()

submission = test_data.copy()


model3.fit(df,label)

submission.columns=["id","label"]

test_comment = pd.DataFrame(index=range(len(test_data)),columns=range(len(featuress[0])))
test_comm = test_data["comment"]


for i in range(len(test_comm)):
    rule=re.compile(r'[ 0-9a-zA-z\P\W]')
    test_comm[i] = np.array(" ".join(jieba.cut(rule.sub('',test_comm[i]))).split())

for i in range(len(test_data)):
    lst = []
    aa = np.array(list(map(lambda x : model.wv[x],test_comm[i])))
    lst = [0]*len(aa[0])
    for j in range(len(aa)):
        lst += aa[j]
    test_comment.iloc[i,:] = lst/len(aa)

submission["label"] = model3.predict(test_comment)

submission.to_csv("submission.csv",index=None)




############################## appendix

model1 = gensim.models.KeyedVectors.load_word2vec_format(model,binary=True)


model = gensim.models.KeyedVectors.load_word2vec_format('word2vec_779845.bin',binary=True)
model.most_similar(u"男人")


for i in range(len(features)):
    strings = []
    for j in range(len(features[i])):
        if features[i][j] not in model.vocab:
                model.add(features[i][j],model.get_vector(features[i][j]),replace=False)
                strings.append(model.word_vec(features[i][j]))
        else:
            strings.append(model.word_vec(features[i][j]))
    features[i] = np.array(strings)




list(map(lambda x : model.word_vec(x),features[0]))

Word2Vec.load("word2vec_779845.bin",binary=True)
for i in range(len(features)):
    features[i] = np.array(list(map(lambda x : model.word_vec(x),features[i])))
list(map(lambda x : model.word_vec(x),features[1]))

for i in range(len(features)):

a = np.array(list(map(lambda x : model.word_vec(x),features[0])))
new_vec = dictionary.doc2bow("男人")
tfidf = models.TfidfModel("男人")
a.all()
model.train([["男人","女人"]],total_examples=model.corpus_count, epochs = 5)
model.most_similar("男人")

model.add('太慢',model.get_vector('太慢'),replace=False)
model
################## Reguler expression
import re
import regex as re
rule=re.compile(r'[ a-zA-z\P\W]')
rule1 = re.compile(r'[^\P{P}-]+')
rule11 = re.compile(r'[\W]')
re.match(rule,features[1])
rule.sub('',features[1])
rule11.sub('',features[1])

string = "Cheerful  sunny  yellow  is  an  attention  getter.  While  it  is  considered  an  optimistic  color,  people  lose  their  tempers  more  often  in  yellow  rooms,  and  babies  will  cry  more.  It  is  the  most  difficult  color  for  the  eye  to  take  in,  so  it  can  be  overpowering  if  overused. "

lst = string.split("  ")
" ".join(lst)










import tensorflow as tf

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()   # 加州房价数据集

housing.data.shape, housing.target.shape

import numpy as np

### 代码开始 ### (≈2 行代码)
X = np.concatenate((housing.data,np.ones((len(housing.target),1))),axis=1)
y = housing.target.reshape(len(housing.target),1)
X = tf.Variable(X)
y = tf.Variable(y)

a = tf.linalg.inv(tf.matmul(X,tf.transpose(X)), adjoint=False, name=None)
b  = tf.transpose(X)
tf.matmul(tf.linalg.inv(tf.matmul(X,tf.transpose(X)), adjoint=False, name=None),tf.transpose(X))


tf.config.list_physical_devices('GPU')



tf.compat.v1.GPUOptions()

tf.linalg.inv(tf.linalg.matmul(X, X, transpose_a=True, transpose_b=False))


import pandas as pd

# 加载数据集
df = pd.read_csv("http://labfile.oss.aliyuncs.com/courses/1211/car.data", header=None)
# 设置列名
df.columns = ['buying', 'maint', 'doors',
              'persons', 'lug_boot', 'safety', 'class']
df.head()



















#
