from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2],[1,4],[1,0],
              [4,2],[4,0],[4,4],
              [4,5],[0,1],[2,2],
              [3,2],[5,5],[1,-1]])
n = int(input())

if n == 0:
    # 使用MiniBatchKMeans方法
    # *****Begin*****#
    kmeans = MiniBatchKMeans(n_clusters=2, batch_size=6,n_init=3)
    kmeans.fit(X)
    # *****End*****#
else:
    # 使用KMeans方法
    # *****Begin*****#
    kmeans = KMeans(n_clusters=2,n_init=3)
    kmeans.fit(X)
    # *****End*****#

# 输出所有点的类别、两类的中心点并预测[0,0],[4,4]的类别
# *****Begin*****#
labels = kmeans.labels_
centers = kmeans.cluster_centers_
prediction = kmeans.predict([[0,0],[4,4]])

print("所有点的类别：", labels)
print("两类的中心点：", centers)
print("[0,0]的类别预测结果：", prediction[0])
print("[4,4]的类别预测结果：", prediction[1])
# *****End*****#



