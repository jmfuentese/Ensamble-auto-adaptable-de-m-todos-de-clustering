import selfAdaptiveClustering
import matplotlib.pyplot as plt, numpy as np, os, pandas as pd, random, warnings
from sklearn import datasets, cluster, metrics
from sklearn.preprocessing import StandardScaler
import DBCV

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def evaluate(X, labels, met=""):
        num_clust = len(np.unique(labels))
        val = 0
        signo = 0
        penalty = 10**9
        #if num_clust > 1:
        if num_clust == 1:
            val = val + penalty
        else:
            if met == "db" or met == "":
                signo = 1
                val = signo*metrics.davies_bouldin_score(X, labels)
            elif met == "calinski":
                signo = -1
                val = signo*metrics.cluster.calinski_harabasz_score(X, labels)
            elif met == "silh":
                signo = -1
                val = signo*metrics.cluster.silhouette_score(X,labels)
            elif met == "dbcv":
                signo = -1
                val = signo*DBCV.DBCV(X, labels)
        return val

def individualPartitions(X, met, IDD, IDE, rfl, PARAMS = [0]):
	X = StandardScaler().fit_transform(X)
	if len(PARAMS) > 1:
		k = np.int_(PARAMS[0])
		eps = PARAMS[1]
		min_samples = np.int_(PARAMS[2])
		results_file_name = "results_same_parameters.csv"
	else:
		k = random.randint(2,20)
		eps = 0.25 + (random.random() * (0.6 - 0.25))
		min_samples = random.randint(2,15)
		results_file_name = "results.csv"
	results_file_name = rfl
	
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
		kmeans = cluster.KMeans(n_clusters=k, random_state=None).fit(X) #random_state = None is not deterministic
		spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="nearest_neighbors").fit(X)
		agglomerative = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward').fit(X)
		dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(X)

	kmeans_labels = kmeans.labels_
	kmeans_score = evaluate(X, kmeans_labels,met)
	spectral_labels = spectral.labels_
	spectral_score = evaluate(X, spectral_labels,met)
	agglomerative_labels = agglomerative.labels_
	agglomerative_score = evaluate(X, agglomerative_labels,met)
	dbscan_labels = dbscan.labels_
	dbscan_labels[dbscan_labels < 0] = 0
	dbscan_score = evaluate(X, dbscan_labels,met)

	string_kmeans_labels = '|'.join(str(e) for e in kmeans_labels)
	string_spectral_labels = '|'.join(str(e) for e in spectral_labels)
	string_agglomerative_labels = '|'.join(str(e) for e in agglomerative_labels)
	string_dbscan_labels = '|'.join(str(e) for e in dbscan_labels)

	with open(results_file_name, "a") as resultsFile:
		s = "\n{},{},KMEANS,{},{},{}, , ,{}".format(IDD,IDE,k,met,kmeans_score,string_kmeans_labels)
		resultsFile.write(s)
		s = "\n{},{},SPECTRAL,{},{},{}, , ,{}".format(IDD,IDE,k,met,spectral_score,string_spectral_labels)
		resultsFile.write(s)
		s = "\n{},{},AGGLOMERATIVE,{},{},{}, , ,{}".format(IDD,IDE,k,met,agglomerative_score,string_agglomerative_labels)
		resultsFile.write(s)
		s = "\n{},{},DBSCAN,{},{},{}, , ,{}".format(IDD,IDE,eps,met,dbscan_score,string_dbscan_labels)
		resultsFile.write(s)


if __name__ == "__main__":
	params = {'Name':['k',      'eps',      'samples'], 
		      'Min':   [2,         0.3,           5    ],
		      'Max':   [21,        0.46,           20   ],
		      'Type':  ['int',   'decimal',      'int' ],
		     }

	mets = ["db", "calinski", "silh"]
	methods = ["hgpa"]

	data_path = "datasets/non-spherical/"
	data_dirs = ["datasets/non-spherical/", "datasets/spherical/"]
	groundTruth_dirs = ["datasets/non-spherical-labels/", "datasets/spherical-labels/"]

	executions = 100
	sample_size_per = .8
	km_params = np.arange(2,21,1)
	same_parameters = True
	results_file_names = ["results_non-spherical.csv", "results_spherical.csv"]
	
	
	for d,directory in enumerate(data_dirs):
		for j,file in enumerate(listdir_nohidden(directory)):
			if "_labels" not in file:
				#try:
				labels = pd.read_csv(groundTruth_dirs[d]+file).values
				expectedK = len(np.unique(labels))
				X = pd.read_csv(directory+file, sep=" ", header=None).iloc[:,:].values
				sample_size = int(sample_size_per*len(X))
				for e in range(executions):
					for metric in mets:
						for method in methods:
							score, solution, best_parameters = selfAdaptiveClustering.SAC(X,params, 15, method, metric, sample_size, 50)
							obtainedK = len(np.unique(solution))
							solution_string = '|'.join(str(e) for e in solution)
							if same_parameters:
								individualPartitions(X,metric,file,e, results_file_names[d], best_parameters)
								results_file_name = "results_same_parameters.csv"
							else:
								individualPartitions(X,metric,file,e, results_file_names[d])
							with open(results_file_names[d], "a") as resultsFile:
								s = "\n{},{},Self-Adaptive,NA,{},{},{},{},{}".format(file,e,metric,score,expectedK,obtainedK,solution_string)
								resultsFile.write(s)
				#except:
				#	print("Error in file: ", file)
				print("PROCESS COMPLETED FOR FILE: ", file)
