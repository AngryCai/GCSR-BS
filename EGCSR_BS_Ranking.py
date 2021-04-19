import numpy as np
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize


class EGCSR_BS_Ranking:

    def __init__(self, n_clusters, regu_coef=1., n_neighbors=10, ro=0.1, save_affinity=False):
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
        # A = A * np.transpose(A)
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
        Coef = self.thrC(C, self.ro)
        Coef = 0.5 * (np.abs(Coef) + np.abs(Coef.T))
        if self.save_affinity:
            np.savez('./model-basic-affinity-ranking.npz', C=C, C1=Coef)
        # Coef = self.thrC(C, self.ro)
        C[np.diag_indices_from(C)] = 0
        C = normalize(C, axis=0)
        return C

    def predict(self, X):
        """
        :param X: shape [n_row*n_clm, n_band]
        :return: selected band subset
        """
        C = self.fit(X)
        selected_band = self.__get_band(C, X)
        return selected_band

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

    def build_aff(self, C):
        N = C.shape[0]
        Cabs = np.abs(C)
        ind = np.argsort(-Cabs, 0)
        for i in range(N):
            Cabs[:, i] = Cabs[:, i] / (Cabs[ind[0, i], i] + 1e-6)
        Cksym = Cabs + Cabs.T
        return Cksym

    def __get_band(self, C, X):
        """
        select band according to the center of each cluster
        :param cluster_result:
        :param X:
        :return:
        """
        C[np.diag_indices_from(C)] = 0
        sum_C = np.linalg.norm(C, axis=1)
        sorted_inx = np.argsort(sum_C)  # ascending order for each column
        largest_k = sorted_inx[-self.n_clusters:]

        # # statistic
        # element, freq = np.unique(largest_k, return_counts=True)
        # selected_inx = element[np.argsort(freq)][-self.n_clusters:]
        print('band index:', largest_k)
        selected_band = X[:, largest_k]
        self.band_indx = largest_k
        return selected_band


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