import time
import warnings
import matplotlib
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sse import compute_sse

np.random.seed(0)

# ============
# 生成数据集时，我们选择足够大的尺寸以观察算法的可扩展性，但又不至于运行时间过长。
# ============
n_samples = 2000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# "各向异性分布的数据" 指的是数据以非均匀或非各向同性的方式分布，也就是数据点的分布在各个方向上并不相等。
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# 生成具有不同方差的 blob 数据集
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# 设置聚类参数
# ============
plt.figure(figsize=(9 * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
# 创建一个数组来存储 SSE 值
sse_values = [0] * 12

def run_clustering(k):
    plot_num = 1

    default_base = {'quantile': .3,
                    'eps': .3,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': k,}

    datasets = [
        (noisy_circles, {'damping': .77, 'preference': -240,
                         'quantile': .2, 'n_clusters': 2,
                         'min_samples': 20, 'xi': 0.25}),
        (noisy_moons, {'damping': .75, 'preference': -220, 'n_clusters': 2}),
        (varied, {'eps': .18, 'n_neighbors': 2,}),
        (aniso, {'eps': .15, 'n_neighbors': 2,}),
        (blobs, {}),
        (no_structure, {})]

    for i_dataset, (dataset, algo_params) in enumerate(datasets):
        # 将dataset中的值更新到"params"变量中
        params = default_base.copy()
        params.update(algo_params)

        X, y = dataset

        # 标准化数据集以便于参数选择
        X = StandardScaler().fit_transform(X)

        # ============
        # 创建聚类对象(cluster objects)
        # ============
        kmeans = cluster.KMeans(n_clusters=params['n_clusters'])# K均值算法
        two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])# Mini-Batch K-Means算法

        clustering_algorithms = (
            ('KMeans', kmeans),
            ('MiniBatchKMeans', two_means),
        )

        for name, algorithm in clustering_algorithms:
            t0 = time.time()

            # 捕获与kneighbors_graph相关的警告信息
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                            "connectivity matrix is [0-9]{1,2}" +
                            " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                            " may not work as expected.",
                    category=UserWarning)
            algorithm.fit(X)

            t1 = time.time()
            y_pred = algorithm.predict(X)
            # 计算 SSE 值
            sse = compute_sse(X, y_pred, algorithm)  # 根据具体聚类算法实现计算 SSE 的函数
            # 将 SSE 值添加到数组中
            # sse_values.append(sse)

            plt.subplot(2, 6, plot_num)

            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # 给离群点添加黑色，如果存在的话
            colors = np.append(colors, ["#000000"])
            plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

            # 绘制每个聚类簇的中心点
            cluster_centers = kmeans.cluster_centers_
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='x', c='black', label='Cluster Centers')

            plt.xlim(-2.5, 2.5)
            plt.ylim(-2.5, 2.5)
            plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                     transform=plt.gca().transAxes, size=15,
                     horizontalalignment='right')
            plt.gca().set_aspect('equal', adjustable='box')
            if plot_num % 2 == 1:
                plt.title('KMeans')
            else:
                plt.title('MiniBatchKMeans')

            sse_values[plot_num - 1] = sse
            plot_num += 1

    plt.show()

sse = [[] for _ in range(4)]  # 创建一个空的四行二维数组
for i in range(2,6):
    run_clustering(i)
    sse[i-2] = sse_values.copy()  # 将 sse_values 的值复制到特定行
print(sse)

x_values = [2, 3, 4, 5]  # 横坐标的值

# 创建一个 2x6 的布局，并画出子图
fig, axs = plt.subplots(2, 6, figsize=(21, 6))

for i in range(2):
    for j in range(6):
        ax = axs[i, j]
        y_values = [sse[k][j+2*i] for k in range(4)]  # 根据索引取出纵坐标的值
        ax.plot(x_values, y_values, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Subplot {i*6+j+1}')

plt.tight_layout()
plt.show()


