import numpy as np
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize


class EGCSR_BS_Clustering:

    def __init__(self, n_clusters, regu_coef=1., n_neighbors=10, ro=0.8, save_affinity=False):
        self.n_clusters = n_clusters
        self.regu_coef = regu_coef
        self.n_neighbors = n_neighbors
        self.ro = ro
        self.save_affinity = save_affinity

    def __adjacent_mat(self, x, n_neighbors=10):
        """
        Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
        A = A * np.transpose(A)
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)
        return normlized_A

    def fit(self, X):
        X_T = np.transpose(X)
        A = self.__adjacent_mat(X_T, self.n_neighbors)
        X_ = np.transpose(X_T)  # shape: n_dim * n_samples
        X_embedding = np.dot(X_, A)
        I = np.eye(X_T.shape[0])
        inv = np.linalg.inv(np.dot(np.transpose(X_embedding), X_embedding) + self.regu_coef * I)
        C = np.dot(np.dot(inv, np.transpose(X_embedding)), X_)
        Coef = self.thrC(C,  self.ro)
        y_pre, C_final = self.post_proC(Coef, self.n_clusters, 3, 18)
        # C_final = 0.5 * (np.abs(C) + np.abs(C.T))
        # spectral = SpectralClustering(n_clusters=self.n_clusters)
        # spectral.fit(C_final)
        # y_pre = spectral.fit_predict(C_final) + 1
        if self.save_affinity:
            np.savez('./model-basic-affinity-clustering.npz', C=C_final, C1=0.5 * (np.abs(C) + np.abs(C.T)))
        return y_pre

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        # r = K *  + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                      assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L

    def predict(self, X):
        """
        :param X: shape [n_row*n_clm, n_band]
        :return: selected band subset
        """
        labels = self.fit(X)
        # print(labels.tolist())
        selected_band = self.__get_band(labels, X)
        return selected_band

    def __get_band(self, cluster_result, X):
        """
        select band according to the center of each cluster
        :param cluster_result:
        :param X:
        :return:
        """
        selected_band = []
        n_cluster = np.unique(cluster_result).__len__()
        # img_ = X.reshape((n_row * n_column, -1))  # n_sample * n_band
        for c in np.unique(cluster_result):
            idx = np.nonzero(cluster_result == c)
            center = np.mean(X[:, idx[0]], axis=1).reshape((-1, 1))
            distance = np.linalg.norm(X[:, idx[0]] - center, axis=0)
            band_ = X[:, idx[0]][:, distance.argmin()]
            selected_band.append(band_)
        bands = np.asarray(selected_band).transpose()
        band_indx = self.get_index(bands, X)
        print(band_indx.tolist())
        self.band_indx = band_indx
        return bands

    def get_index(self, selected_band, raw_HSI):
        """
        :param selected_band: 3-D cube
        :param raw_HSI: 3-D cube
        :return:
        """
        band_index = []
        for i in range(selected_band.shape[-1]):
            band_i = np.reshape(selected_band[:, i], (selected_band.shape[0], 1))
            band_index.append(np.argmin(np.sum(np.abs(raw_HSI - band_i), axis=0)))
        return np.asarray(band_index)

"""
import sklearn.datasets as dt
from sklearn import preprocessing
from Toolbox.Preprocessing import Processor
from sklearn.metrics import accuracy_score

p = Processor()
X, y = dt.load_iris(return_X_y=True)
y = p.standardize_label(y)
X = preprocessing.normalize(X)

model = HyperGCSC(n_clusters=3, regu_coef=1e2, n_neighbors=20)
y_pre = model.fit(X)
acc = model.cluster_accuracy(y, y_pre)
print('acc==>', acc)

sc = SpectralClustering(n_clusters=3)
sc_y_pre = sc.fit_predict(X)
sc_acc = model.cluster_accuracy(y, sc_y_pre)
print('acc==>', sc_acc)
"""