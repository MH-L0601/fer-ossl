"""图聚类（谱聚类）工具。

用于在特征空间上进行无监督聚类，替代线性头 argmax 作为 GCD / Debiased
分支的聚类预测。谱聚类先构建 k-NN 相似图，再对图拉普拉斯做谱分解，最后对
谱嵌入做 KMeans，能更好地捕捉特征空间的流形结构。

同时提供 kmeans 作为备选（直接对特征做 KMeans）。
"""
import numpy as np


def graph_cluster(features, n_clusters, k=20, seed=0, n_init=10, n_jobs=-1):
    """基于 k-NN 图的谱聚类。

    Args:
        features: [N, D] 特征矩阵（建议先做 L2 归一化，使欧氏距离等价于余弦相似度）。
        n_clusters: 聚类数量（= 总类别数，RAF-DB 为 7）。
        k: k-NN 图的邻居数。
        seed: 随机种子。
        n_init: 谱嵌入后 KMeans 的初始化次数。
        n_jobs: KMeans 并行数（-1 表示使用全部核）。

    Returns:
        labels: [N] int64 聚类标签 (0..n_clusters-1)。
    """
    from sklearn.cluster import SpectralClustering

    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    if n < n_clusters:
        raise ValueError(f'[graph_cluster] n_samples({n}) < n_clusters({n_clusters})')

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=min(k, n - 1),
        random_state=seed,
        assign_labels='kmeans',
        n_init=n_init,
        n_jobs=n_jobs,
    )
    return sc.fit_predict(features).astype(np.int64)


def kmeans_cluster(features, n_clusters, seed=0, n_init=10):
    """直接对特征做 KMeans（备选聚类方式）。"""
    from sklearn.cluster import KMeans

    features = np.asarray(features, dtype=np.float64)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    return km.fit_predict(features).astype(np.int64)


def gmm_cluster(features, n_clusters, seed=0, covariance_type='full', n_init=5):
    """高斯混合模型聚类（备选聚类方式）。

    与 KMeans 不同，GMM 允许某个分量成为宽大的高斯，从而用单个宽簇
    捕获占主导的类别（如 FER 中占多数的 happiness），而不是像 KMeans
    那样把主导类拆成多个小簇。

    Args:
        covariance_type: 'full'（椭圆簇，O(d^3) 每分量）或 'diag'
            （对角协方差，O(d)，可直接在 256 维上运行）。
    """
    from sklearn.mixture import GaussianMixture

    features = np.asarray(features, dtype=np.float64)
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=covariance_type,
        init_params='kmeans',
        reg_covar=1e-4,
        max_iter=200,
        n_init=n_init,
        random_state=seed,
    )
    return gmm.fit_predict(features).astype(np.int64)


def cluster_features(features, n_clusters, method='graph', k=20, seed=0, n_init=10, n_jobs=-1):
    """统一的聚类入口。

    Args:
        method: 'graph'（谱聚类，默认）、'kmeans'、'gmm'（full 协方差）或 'gmm_diag'（对角协方差）。
    """
    features = np.asarray(features, dtype=np.float64)
    method = (method or 'graph').lower()
    if method == 'graph':
        return graph_cluster(features, n_clusters, k=k, seed=seed, n_init=n_init, n_jobs=n_jobs)
    if method == 'kmeans':
        return kmeans_cluster(features, n_clusters, seed=seed, n_init=n_init)
    if method == 'gmm':
        return gmm_cluster(features, n_clusters, seed=seed, covariance_type='full', n_init=max(n_init, 1))
    if method == 'gmm_diag':
        return gmm_cluster(features, n_clusters, seed=seed, covariance_type='diag', n_init=max(n_init, 1))
    raise ValueError(f"Unknown clustering method '{method}'; expected 'graph', 'kmeans', 'gmm' or 'gmm_diag'")
