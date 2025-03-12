import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 初始化并训练逻辑回归模型
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 可视化：绘制决策边界（仅适用于两个特征的情况）
# 为了简单起见，我们选择前两个特征进行可视化
X_train_vis = X_train[:, :2]
X_test_vis = X_test[:, :2]

# 创建一个网格来绘制决策边界
x_min, x_max = X_train_vis[:, 0].min() - 1, X_train_vis[:, 0].max() + 1
y_min, y_max = X_train_vis[:, 1].min() - 1, X_train_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 预测网格上每个点的类别
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界和训练样本
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X_train_vis[:, 0], X_train_vis[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired, s=20)
plt.scatter(X_test_vis[:, 0], X_test_vis[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired, marker='x', s=50)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.title('Logistic Regression Decision Boundary (Iris Dataset - First Two Features)')
plt.legend(['Decision Boundary', 'Train Set', 'Test Set'], loc='upper left')
plt.show()