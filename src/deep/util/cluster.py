from sklearn import cluster
import numpy
# import util.affinity_matrix
def spectral(dataset, n_cluster, matrix_type = "data"):
    """
    matrix_type has two value: data or affinity
    
    """
    feature_matrix = numpy.asarray(dataset)
#     matrix_type = "affinity"
#     matrix_type = "data"
    if matrix_type == "affinity":
        print "Calculate affinity."
#         affinty_matrix = util.affinity_matrix.compute_affinity_matrix(feature_matrix)[0]
        affinty_matrix = feature_matrix
        print "Calculated affinity. Start to cluster."
        
        spectral = cluster.SpectralClustering(n_clusters=n_cluster,
                                          eigen_solver='arpack',
                                          affinity="precomputed")
        spectral.fit(affinty_matrix)
    elif matrix_type == "data":
        spectral = cluster.SpectralClustering(n_clusters=n_cluster,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
        spectral.fit(dataset)
    cluster_labels = spectral.labels_.astype(numpy.int)
    print "Clustering finished."
    return cluster_labels

def kmeans(dataset, n_cluster = 625):
    """
    Cluster the given data set into n_cluster parts.
    """
    from scipy.cluster.vq import kmeans2, whiten
    feature_matrix = numpy.asarray(dataset)
    whitened = whiten(feature_matrix)
    cluster_num = 625
    _, cluster_labels = kmeans2(whitened, cluster_num, iter = 100)
    return cluster_labels