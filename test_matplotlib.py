from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import graphviz
from matplotlib import pyplot as plt


# 导入数据，分训练测试集
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(wine['data'], wine['target'], test_size=0.3)

# 最大深度学习曲线
score_list = []

for i in range(1, 11):
    clf = tree.DecisionTreeClassifier(max_depth=i)
    clf = clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    score_list.append(score)

print(score_list)
plt.plot(range(1, 11), score_list, label='max_depth', color='blue')
plt.legend()
plt.show()
