"""
This is a simulation problem. You should create a data with at least 4 clusters of different sizes
(number of observations), shapes or spread. For questions (a) and (b), the focus is on comparing
clustering methods and cluster number selection methods for data where the clusters are of the
same size. For questions (c) and (d) you will investigate the setting where cluster sizes differ. Make
sure you discuss and compare the outcome.

2(a)
Start with a balanced cluster size but clusters have different spread or shape. Use at least 2 different
clustering methods and investigate clustering accuracy.

2(b)
Investigate at least 2 approaches to selecting the number of clusters in the data set.

2(c)
Now simulate a data set where the clusters contain different number of observations and repeat
task (a).
2(d)

For the data in (c), repeat task (b).
"""

from cProfile import label
from matplotlib import pyplot as plt
import numpy as np
import numpy.random as rnd
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, adjusted_rand_score, homogeneity_score, completeness_score
from scipy.cluster.hierarchy import dendrogram, linkage

rnd.seed(335)

# Construc a data set with at least 4 clusters of different sizes

#generate 4 gausian clusters


def generate_clusters(n_clusters, n_obs, n_dim, random_num_obs = False):
    """
    Generate a data set with n_clusters clusters of size n_obs and n_dim dimensions.
    :param n_clusters: number of clusters
    :param n_obs: number of observations per cluster
    :param n_dim: number of dimensions
    :return: data set
    """

    data = np.zeros((n_clusters * n_obs, n_dim))
    labels = np.zeros((n_clusters * n_obs))

    for i in range(n_clusters):
        # Random covariance matrix
        cov = np.random.rand(n_dim, n_dim)*2
        data[i * n_obs:(i + 1) * n_obs, :] = rnd.multivariate_normal(np.zeros(n_dim), cov, n_obs) + i*4
        labels[i * n_obs:(i + 1) * n_obs] = i
    return data, labels


def generate_clusters_random_num_sampels(n_clusters, n_dim):
    """
    Generate a data set with n_clusters clusters of size n_obs and n_dim dimensions.
    """
    num_sampels = np.random.randint(1, 200, size=n_clusters)
    data = np.zeros((num_sampels.sum(), n_dim))
    labels = np.zeros((num_sampels.sum()))

    for i in range(n_clusters):
        # Random covariance matrix
        num_samp = num_sampels[i]
        current_index = num_sampels[:i].sum()
        cov = np.random.rand(n_dim, n_dim)*2
        data[current_index:current_index+num_samp, :] = rnd.multivariate_normal(np.zeros(n_dim), cov, num_samp) + i*4
        labels[current_index:current_index+num_samp] = i

    print(num_sampels)
    return data, labels


if __name__ == '__main__':
    # generate 4 clusters
    data, labels = generate_clusters(4, 100, 2)
    # plot data

    #cluster with kmeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
    plt.title('Kmeans, ' + f"Adjusted_Rand_Score: {adjusted_rand_score(labels, kmeans.labels_)}")
    plt.savefig('KmeanClusterSameSize.png')
    plt.cla()

    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward').fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=agg.labels_)
    plt.title('Agglomerative, ' + f"Adjusted_Rand_Score: {adjusted_rand_score(labels, agg.labels_)}")
    plt.savefig('AgglomerativeClusterSameSize.png')
    plt.cla()

    #Unsupervised Hierarchical clustering
    
    ap_clustering = AffinityPropagation().fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=ap_clustering.labels_)
    plt.title('AffinityPropagation, ' + f"Adjusted_Rand_Score: {adjusted_rand_score(labels, ap_clustering.labels_)}")
    plt.savefig('AffinityPropagationClusterSameSize.png')
    plt.cla()

    GaussianMixture(n_components=4, random_state=0).fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=GaussianMixture(n_components=4, random_state=0).fit(data).predict(data))
    plt.title('GaussianMixture, ' + f"Adjusted_Rand_Score: {adjusted_rand_score(labels, GaussianMixture(n_components=4, random_state=0).fit(data).predict(data))}")
    plt.savefig("GaussianMixtureClusterSameSize.png")
    plt.cla()

    data, labels = generate_clusters_random_num_sampels(4, 2)

    #cluster with kmeans
    kmeans = KMeans(n_clusters=4, random_state=0).fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_)
    plt.title('Kmeans, ' + f"Adjusted_Rand_Score: {adjusted_rand_score(labels, kmeans.labels_)}")
    plt.savefig('unbalance_cluster_kmean.png')
    plt.cla()

    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward').fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=agg.labels_)
    plt.title('Agglomerative, ' + f"Adjusted_Rand_Score: {adjusted_rand_score(labels, agg.labels_)}")
    plt.savefig('unbalance_cluster_agglomerative.png')
    
    ap_clustering = AffinityPropagation().fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=ap_clustering.labels_)
    plt.title('AffinityPropagation, ' + f"Adjusted_Rand_Score: {adjusted_rand_score(labels, ap_clustering.labels_)}")
    plt.savefig('unbalance_cluster_affinity_propagation.png')
    plt.cla()

    GaussianMixture(n_components=4, random_state=0).fit(data)
    plt.scatter(data[:, 0], data[:, 1], c=GaussianMixture(n_components=4, random_state=0).fit(data).predict(data))
    plt.title('GaussianMixture, ' + f"Adjusted_Rand_Score: {adjusted_rand_score(labels, GaussianMixture(n_components=4, random_state=0).fit(data).predict(data))}")
    plt.savefig("unbalance_cluster_gaussian_mixture.png")
    plt.cla()