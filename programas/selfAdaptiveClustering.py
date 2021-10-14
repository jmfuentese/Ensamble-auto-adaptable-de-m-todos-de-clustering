""" ----------------------------------------------------------------------
    The python code for a self-adaptive ensemble of clustering algorithms 
    ----------------------------------------------------------------------
    Input:
      X: data
      method: consensus function to combine individual partitions
      met: validity index to evaluate the combined partition
    Output:
      solution: a combined partition
    ----------------------------------------------------------------------
    Marco Fuentes
    Cinvestav Tamaulipas
"""
import sklearn, matplotlib.pyplot as plt, numpy as np, random, pandas as pd, time, warnings, sys, math, warnings
import pymetis
from sklearn import datasets, metrics, cluster
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.stats import mode
import DBCV
from s_dbw import S_Dbw

class Model():
    def configure(params = [2, 0.3, 2]):
        p_K, p_eps, p_samples = params
        if p_K < 2: p_K = 2
        p_K = np.int_(p_K)
        p_samples = np.int_(p_samples)
        kmeans = cluster.KMeans(n_clusters=p_K, random_state=None) #random_state = None is not deterministic
        spectral = cluster.SpectralClustering(n_clusters=p_K, 
                                              eigen_solver='arpack', 
                                              affinity="nearest_neighbors")
        agglomerative = cluster.AgglomerativeClustering(n_clusters=p_K, 
                                                affinity='euclidean', 
                                                linkage='ward')
        dbscan = cluster.DBSCAN(eps=p_eps, min_samples=p_samples)
    
        clustering_algorithms = (dbscan, spectral, agglomerative, kmeans)
        return clustering_algorithms


    def sampling_data(X, S):
        T = []
        for i in range(S):
            T.append(X.loc[X['index']%S == i])
        return T
    
    def fit(clustering_algorithms, X):
        solutions = []
        for algorithm in clustering_algorithms:
            t0 = time.time()

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

            labels_pred = np.array(algorithm.labels_)
            #y_pred = algorithm.predict(X)
            if len(np.unique(labels_pred)) >= 2:
                solutions.append(labels_pred)
        return np.array(solutions)

    def evaluation(X, labels, met=""):
        #try:
        num_clust = len(np.unique(labels))
        val = 0
        signo = 0
        penalty = 10**9
        #if num_clust > 1:
        if met == "db" or met == "":
            signo = 1
            if num_clust == 1:
                val = val + penalty
            else:
                val = signo*metrics.davies_bouldin_score(X, labels)
        elif met == "calinski":
            signo = -1
            if num_clust == 1:
                val = val + penalty
            else:
                val = signo*metrics.cluster.calinski_harabasz_score(X, labels)
        elif met == "silh":
            signo = -1
            if num_clust == 1:
                val = val + penalty
            else:
                val = signo*metrics.cluster.silhouette_score(X,labels)
        elif met == "dbcv":
            signo = -1
            if num_clust == 1:
                val = val + penalty
            else:
                val = signo*DBCV.DBCV(X, labels)
        elif met == "s_dbw":
            signo = 1
            if num_clust == 1:
                val = val + penalty
            else:
                val = signo*S_Dbw(X, labels, centers_id=None, method='Tong', alg_noise='bind',
                    centr='mean', nearest_centr=True, metric='euclidean')
        #else:
        #    val = 10**9
        #except: 
        #    return -1
        return val


    def display(self, X, solutions):
        algorithm_names = ['KMeans', 'SpectralClustering', 'AgglomerativeClustering', 'DBSCAN']
        f, axarr = plt.subplots(1,4,figsize=(24,6))
        i = 0
        for i, solution in enumerate(solutions):
            axarr[i].scatter(X[:,0],  X[:,1], c=solution)
            axarr[i].title.set_text(algorithm_names[i])
            i+=1
        plt.show()

class Voting(Model):
    def __init__(self, met):
        self.met = met
        self.algorithms = Model.configure()

    def update(self, params):
        self.algorithms = Model.configure(params)

    def fit(self, X):
        solutions = Model.fit(self.algorithms, X)

        relabeled_solutions = []
        relabeled_solutions.append(solutions[0])
        for i, solution in enumerate(solutions[:-1]):
            relabeled_solutions.append(self.label_matching(solutions[i+1], solutions[0]))
        consensus_solution = self.consensus(relabeled_solutions)
        if -1 in consensus_solution:
            consensus_solution[consensus_solution == -1] = max(np.unique(consensus_solution))
        try:
            consensus_score = Model.evaluation(X, consensus_solution, self.met)
        except:
            consensus_score = 10**9
            #print("Error. ")
            #exit()

        return consensus_score, consensus_solution

    def consensus(self, solutions):
        consensus_solution = np.zeros_like(solutions[0])
        for i in range(len(consensus_solution)):
            elements = []
            for j in range(len(solutions)):
                elements.append(solutions[j][i])
                consensus_solution[i] = mode(elements)[0]
        #consensus_solution = np.zeros_like(solutions[0])
        #for i in range(0, len(consensus_solution)):
        #    consensus_solution[i] = mode((solutions[0][i], solutions[1][i], solutions[2][i], solutions[3][i]))[0]
        return consensus_solution

    def label_matching(self,labels,reference):
        new_labels = np.zeros_like(reference)
        new_temp = np.ones_like(reference)
        while np.sum(new_labels == reference) < len(reference):
            new_temp = new_labels
            for i in range(len(reference)):
                mask=(labels==i)
                new_labels[mask]=mode(reference[mask])[0]
            if (new_temp == new_labels).all():
                return new_labels
        else:
            return new_labels


class Coassoc(Model):
    def __init__(self, K):
        self.algorithms = Model.configure(K)

    def fit(self, X):
        pass


class HybridBipartiteGraph(Model):
    def __init__(self, met):
        self.met = met
        self.algorithms = Model.configure()

    def update(self, params):
        self.algorithms = Model.configure(params)

    def fit(self, X):
        solutions = Model.fit(self.algorithms, X)

        consensus_solution = self.hgpa(solutions, 111)
        if -1 in consensus_solution:
            consensus_solution[consensus_solution == -1] = max(np.unique(consensus_solution))
        try:
            consensus_score = Model.evaluation(X, consensus_solution, self.met)
        except:
            consensus_score = 10**9
        return consensus_score, consensus_solution

    def create_hypergraph(self, base_clusters):
        H = None
        len_bcs = base_clusters.shape[1]

        for bc in base_clusters:
            unique_bc = np.unique(bc[~np.isnan(bc)])
            len_unique_bc = len(unique_bc)
            bc2id = dict(zip(unique_bc, np.arange(len_unique_bc)))
            h = np.zeros((len_bcs, len_unique_bc), dtype=int)
            for i, elem_bc in enumerate(bc):
                if not np.isnan(elem_bc):
                    h[i, bc2id[elem_bc]] = 1
            if H is None:
                H = h
            else:
                H = np.hstack([H, h])
        return H

    def to_pymetis_format(self,adj_mat):
        xadj = [0]
        adjncy = []
        eweights = []

        for row in adj_mat:
            idx = np.nonzero(row)[0]
            val = row[idx]
            adjncy += list(idx)
            eweights += list(val)
            xadj.append(len(adjncy))
        
        return xadj, adjncy, eweights

    def hgpa(self, base_clusters, random_state):
        nclass = len(np.unique(base_clusters[0]))
        A = self.create_hypergraph(base_clusters)
        rowA, colA = A.shape
        W = np.vstack([np.hstack([np.zeros((colA, colA)), A.T]), np.hstack([A, np.zeros((rowA, rowA))])])
        xadj, adjncy, _ = self.to_pymetis_format(W)
        membership = pymetis.part_graph(nparts=nclass, xadj=xadj, adjncy=adjncy, eweights=None)[1]
        celabel = np.array(membership[colA:])
        return celabel


class IM(Model):
    def __init__(self, K):
        self.algorithms = Model.configure(K)

    def fit(self, X):
        pass

class Ensemble():
    @staticmethod
    def get_method(method, met):
        try:
            if method == "voting":
                return Voting(met)
            elif method == "coassoc":
                return Coassoc(K)
            elif method == "hgpa":
                return HybridBipartiteGraph(met)
            elif method == "im":
                return IM(K)
            raise AssertionError("Method not found")
        except AssertionError as _e:
            print(_e)
            

class ParamsDifferentialAdapter:
    def __init__(self,variablesSetting, cr=0.9, nu=0.09, n=10):
        self.cr=cr #Differential evolution parameter
        self.nu=nu #Differential evolution parameter
        self.variablesSetting=variablesSetting
        self.num_params=len(variablesSetting.index)# Number of parameters
        self.n=n
        self.P=np.zeros((self.n,self.num_params))# Population
        self.F=np.zeros(self.n); #Fitness of all individuals
        
    def initialization(self):
        for i in range(self.n):
            for j in range(self.num_params):
                bounds=self.variablesSetting.loc[j,["Min", "Max"]]
                paramValue=np.random.uniform (bounds[0],bounds[1])
                self.P[i,j]=self.setDomainValue(j,paramValue)
            
    def setDomainValue(self, idParam, paramValue):
        param_type=self.variablesSetting.loc[idParam,"Type"]
        bounds=self.variablesSetting.loc[idParam,["Min", "Max"]]
        if bounds[0] >= 0 and paramValue < 0:
            paramValue = paramValue*-1
        if (param_type=="decimal"):
            return paramValue
        else:
            return np.int_(paramValue)

    def getParams (self,index):
        return self.P[index]
    
    def setFitness (self, id, value):
        self.F[id]=value
        
    def getFitness (self, id):
        return self.F[id]

    def recombine(self, id):
        candidates_index= np.floor(np.random.uniform(0,self.n-1,3))
        candidates_index= candidates_index.astype(int)
        z_j=np.zeros(self.num_params)
        for i in range(self.num_params):
            bounds=self.variablesSetting.loc[i,["Min", "Max"]]
            min=bounds[0]
            max=bounds[1]
            i_rand =math.floor(np.random.uniform(1,self.num_params,1))
            v_i = self.P[candidates_index[0],i]+(self.nu*(self.P[candidates_index[1],i]-self.P[candidates_index[2],i]))
            in_the_domain=(min <= v_i) and (v_i <= max)
            z_j[i] = v_i if (np.random.uniform(0,1,1)<=self.cr or i==i_rand) and in_the_domain else self.P[id,i]
            z_j[i] = self.setDomainValue(i, z_j[i])
      
        return z_j #Individual updated

    def sortPopulation(self):
        indices = np.argsort(self.F)
        self.F=self.F[indices]
        self.P=self.P[indices]

def SAC(X,params,population,consensus_method,metric,sample_size, iterations):
    variablesSetting = pd.DataFrame(params)

    #prepare adapter
    adapter=ParamsDifferentialAdapter(variablesSetting, population)
    adapter.initialization()

    X = StandardScaler().fit_transform(X)

    #Instantiate the ensemble
    ensemble = Ensemble.get_method(consensus_method, metric)

    #First stage: Evaluate initial population
    for i in range(adapter.n):
        data=resample(X, n_samples=sample_size, replace=True)
        ensemble.update(adapter.P[i])
        evaluation_i = ensemble.fit(data)[0]
        adapter.setFitness(i, evaluation_i)

    #Second stage: Iterative Adaptation of parameters 
    for r in range(iterations):
        data=resample(X, n_samples=sample_size, replace=True)
        for i in range(adapter.n): 
          #Update the i-th individual
          z=adapter.recombine(i)
          #Evaluation
          ensemble.update(z)
          f_z=ensemble.fit(data)[0]
          #IF max: f_z=f_z*-1
          if (f_z<adapter.F[i]):
                adapter.P[i]=z
                adapter.setFitness(i, f_z)
        s = "Iteration {} Complete".format(r+1)
        print(s)
    adapter.sortPopulation()

    #Final stage: update the ensemble with the best parameters set and fit
    best_parameters = adapter.P[0]
    ensemble.update(best_parameters)
    score, solution = ensemble.fit(X)

    return score, solution, best_parameters
