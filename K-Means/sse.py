import numpy as np

def compute_sse(X, y_pred, algorithm):
    centers = algorithm.cluster_centers_  # 获取聚类中心点
    labels = algorithm.labels_  # 获取聚类标签

    sse = 0  # 初始化 SSE 值

    # 遍历每个样本
    for i in range(len(X)):
        # 计算样本与对应簇中心点之间的欧几里德距离
        distance = np.linalg.norm(X[i] - centers[labels[i]])
        # 累加距离的平方到 SSE 值中
        sse += distance ** 2

    return sse

