import os
import sys
import subprocess

os.chdir(rf"{os.path.realpath(os.path.dirname(__file__))}")
# subprocess.check_call([sys.executable, 'pip', 'install', '-r', './requirements.txt'])

import requests
import nltk
import pandas as pd
import numpy as np
import io
import itertools
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
nltk.download('stopwords')
from pandas.core.frame import DataFrame
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import cluster
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.elbow import elbow
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

def saveList(myList,filename):
    np.save(filename,myList)
    print("Saved successfully!")

def loadList(filename):
    tempNumpyArray=np.load(filename, allow_pickle='TRUE')
    return tempNumpyArray.tolist()

def kMeansClusteringEvaluation(nClusters: int, data, params: list, filename: str):
    noClust = []
    silCo = []
    algUsed = []
    CalHar = []
    DavBoul = []
    print("======================================================")
    print("===CALCULATING K-MEANS CLUSTERING EVALUATION SCORES===")
    print("======================================================\n")

    for algo in params:
        km = KMeans(n_clusters=nClusters, algorithm=algo)
        labels = km.fit_predict(data)

        noClust.append(nClusters)
        algUsed.append(algo)
        silCo.append(silhouette_score(data, labels))
        CalHar.append(calinski_harabasz_score(data, labels))
        DavBoul.append(davies_bouldin_score(data, labels))
        print(f"{algo} evaluations score done !!!")

    df_kmean_evaluation = pd.DataFrame(
        {
            'Cluster Number': noClust,
            'Algoritm Used': algUsed,
            'Silhouette Score': silCo,
            'Calinski-Harabasz Score': CalHar,
            'Davies-Bouldin Score': DavBoul
        }
    )
    df_kmean_evaluation.to_csv(f"./KMeans/{filename}.csv", index=False)
    print(f"\nsaving {filename}.csv...")
    print(f"{filename}.csv saved successfully!\n")

def kMeansClustering(
    rmvDataset: DataFrame,
    dbDataset: DataFrame,
    mainDataset: DataFrame,
    nClusters: int,
    data,
    params: list,
):
    print("======================================================")
    print("========PERFORMING K-MEANS CLUSTERING ALGORITHM=======")
    print("======================================================\n")
    for algo in params:
        df_disease = pd.DataFrame(mainDataset['Disease'])
        df_disease = df_disease.drop_duplicates(subset=['Disease'])
        df_disease.reset_index(drop=True, inplace=True)

        model = KMeans(n_clusters=nClusters, algorithm=algo)
        model.fit(data)
        df_disease['cluster'] = model.labels_

        clusterList = []
        diseaseList = []
        chemicalList = []
        corduidList = []
        chemCountlist = []
        uidCountlist = []
        includeList = []
        inCountlist = []
        percentageList = []
        print(f"Clustering for {algo} parameter...")
        for clusternumber in range(nClusters):
            Alist = []
            Blist = []
            Clist = []
            Dlist = []
            Elist = []
            Flist = []
            clusterList.append(clusternumber)
            listDis = list(df_disease['Disease'][df_disease['cluster'] == clusternumber])
            diseaseList.append(listDis)
            listChem = []
            cordUID = []
            for disease in listDis:
                for chem in list(mainDataset['Chemical'][mainDataset['Disease'] == disease].values):
                    listChem.append(chem)
                    # longList = list(mainDataset['Cord_UID_List'][mainDataset['Relations'] == str(disease+' --> '+chem)].values)[0].split("'")
                    # for uid in longList:
                    #     if len(uid) > 2:
                    #         cordUID.append(uid)
            listChem = list(set(listChem))
            Alist = listChem
            for filtchem in Alist:
                if filtchem not in list(rmvDataset['compounds'].values):
                    Clist.append(filtchem)
            for dis in listDis:
                for supchem in list(mainDataset['Chemical'][mainDataset['Disease'] == dis].values):
                    if supchem in Clist:
                        if mainDataset['Count'][mainDataset['Relations'] == str(dis+' --> '+supchem)].values >= 3:
                            Dlist.append(supchem)
                            longList = list(mainDataset['Cord_UID_List'][mainDataset['Relations'] == str(dis+' --> '+supchem)].values)[0].split("'")
                            for uid in longList:
                                if len(uid) > 2:
                                    cordUID.append(uid)
            Elist = intersection(Dlist, list(dbDataset['DRUG']))
            try:
                Flist = (len(Elist)*100)/len(Dlist)
                percentageList.append(str(str(round(Flist,3))+'%'))
            except ZeroDivisionError:
                Flist = "No chemicals/drugs with sufficient minimum support for this cluster"
                percentageList.append(Flist)
            chemicalList.append(Dlist)
            cordUID = list(set(cordUID))
            corduidList.append(cordUID)
            chemCountlist.append(len(Dlist))
            uidCountlist.append(len(cordUID))
            includeList.append(Elist)
            inCountlist.append(len(Elist))
            # result = (len(intersection(listChem, list(dbDataset['DRUG'])))*100)/len(list(dbDataset['DRUG']))
            # percentageList.append(str(str(round(result,2))+'%'))
        df_result = pd.DataFrame(
            {
                'Cluster Number': clusterList,
                'List of Disease': diseaseList,
                'List of Chemical': chemicalList,
                'List of CORD UID': corduidList,
                'Chemical Count': chemCountlist,
                'CORD UID Count': uidCountlist,
                'List of Chemicals in Drugbank': includeList,
                'Chemicals in Drugbank Count': inCountlist,
                'Persentage': percentageList
            }
        )
        df_result = df_result.sort_values('CORD UID Count')
        df_result.to_csv(f"./KMeans/KMeans-Clustering-Result({algo}).csv", index=False)
        print(f"KMeans-Clustering-Result({algo}).csv saved successfully!")
        df_disease.to_csv(f"./KMeans/KMeans-Clustering-Corpus({algo}).csv", index=False)
        print(f"KMeans-Clustering-Corpus({algo}).csv saved successfully!\n")

clusters = []

def agglClusteringEvaluation(
    data,
    method: list,
    distance: list,
    linkage: list,
    filename: str
    ):
    global clusters
    noClust = []
    parameters = []
    silCo = []
    CalHar = []
    DavBoul = []
    matplotlib.use('Agg')
    print("======================================================")
    print("CALCULATING AGGLOMERATIVE CLUSTERING EVALUATION SCORES")
    print("======================================================\n")

    for params in method:
        print(f"Performing Dendrogram for method '{params}'")
        plt.figure(figsize=(10,7))
        plt.title(f"Dendrogram for method='{params}'")
        dend = shc.dendrogram(shc.linkage(data, method=params))
        plt.savefig(f"./Agglomerative/dendrogram({params}).png")
        print(f"dendrogram({params}).png saved successfully!")
        print(f"{len(list(set(dend['color_list'])))} Clusters Acquired from '{params}' method\n")
        clusters.append(len(list(set(dend['color_list']))))
    clusters = list(set(clusters))
    print(f"Best Clusters for Agglomerative Clustering\t: {clusters}\n")

    for k in clusters:
        for clustarg in list(itertools.product(distance,linkage)):
            if clustarg[1] != 'ward':
                print(f"Calculating Evaluation Score for affinity='{clustarg[0]}', linkage='{clustarg[1]}'")
                noClust.append(k)
                parameters.append(str(clustarg[0]+'-'+clustarg[1]))
                km = AgglomerativeClustering(n_clusters=k, affinity=clustarg[0], linkage=clustarg[1])
                labels = km.fit_predict(data)

                silCo.append(round(silhouette_score(data, labels), 6))
                CalHar.append(round(calinski_harabasz_score(data, labels), 6))
                DavBoul.append(round(davies_bouldin_score(data, labels), 6))
            elif clustarg[1] == 'ward' and clustarg[0] == 'euclidean':
                print(f"Calculating Evaluation Score for affinity='{clustarg[0]}', linkage='{clustarg[1]}'")
                noClust.append(k)
                parameters.append(str(clustarg[0]+'-'+clustarg[1]))
                km = AgglomerativeClustering(n_clusters=k, affinity=clustarg[0], linkage=clustarg[1])
                labels = km.fit_predict(data)

                silCo.append(round(silhouette_score(data, labels), 6))
                CalHar.append(round(calinski_harabasz_score(data, labels), 6))
                DavBoul.append(round(davies_bouldin_score(data, labels), 6))
    
    df_aggl_evaluation = pd.DataFrame(
        {
            'Cluster Number': noClust,
            'Clustering Parameters': parameters,
            'Silhouette Score': silCo,
            'Calinski-Harabasz Score': CalHar,
            'Davies-Bouldin Score': DavBoul
        }
    )

    print(f"\nsaving {filename}.csv...")
    df_aggl_evaluation.to_csv(f"./Agglomerative/{filename}.csv", index=False)
    print(f"{filename}.csv saved successfully!\n")

def agglClustering(
    data,
    rmvDataset: DataFrame,
    dbDataset: DataFrame,
    mainDataset: DataFrame,
    distance: list,
    linkage: list
    ):
    global clusters
    print("======================================================")
    print("=====PERFORMING AGGLOMERATIVE CLUSTERING ALGORITHM====")
    print("======================================================\n")
    
    for k in clusters:
        for clustarg in list(itertools.product(distance,linkage)):
            if clustarg[1] != 'ward':
                print(f"Performing Agglomerative Clustering for affinity='{clustarg[0]}', linkage='{clustarg[1]}'")
                df_disease = pd.DataFrame(mainDataset['Disease'])
                df_disease = df_disease.drop_duplicates(subset=['Disease'])
                df_disease.reset_index(drop=True, inplace=True)
                model = AgglomerativeClustering(n_clusters=k, affinity=clustarg[0], linkage=clustarg[1])
                model.fit(data)
                df_disease['cluster'] = model.labels_

                clusterList = []
                diseaseList = []
                chemicalList = []
                corduidList = []
                chemCountlist = []
                uidCountlist = []
                includeList = []
                inCountlist = []
                percentageList = []

                for clusternumber in range(k):
                    Alist = []
                    Blist = []
                    Clist = []
                    Dlist = []
                    Elist = []
                    Flist = []
                    clusterList.append(clusternumber)
                    listDis = list(df_disease['Disease'][df_disease['cluster'] == clusternumber])
                    diseaseList.append(listDis)
                    listChem = []
                    cordUID = []
                    for disease in listDis:
                        for chem in list(mainDataset['Chemical'][mainDataset['Disease'] == disease].values):
                            listChem.append(chem)
                            # longList = list(mainDataset['Cord_UID_List'][mainDataset['Relations'] == str(disease+' --> '+chem)].values)[0].split("'")
                            # for uid in longList:
                            #     if len(uid) > 2:
                            #         cordUID.append(uid)
                    listChem = list(set(listChem))
                    Alist = listChem
                    for filtchem in Alist:
                        if filtchem not in list(rmvDataset['compounds'].values):
                            Clist.append(filtchem)
                    for dis in listDis:
                        for supchem in list(mainDataset['Chemical'][mainDataset['Disease'] == dis].values):
                            if supchem in Clist:
                                if mainDataset['Count'][mainDataset['Relations'] == str(dis+' --> '+supchem)].values >= 3:
                                    Dlist.append(supchem)
                                    longList = list(mainDataset['Cord_UID_List'][mainDataset['Relations'] == str(dis+' --> '+supchem)].values)[0].split("'")
                                    for uid in longList:
                                        if len(uid) > 2:
                                            cordUID.append(uid)
                    Elist = intersection(Dlist, list(dbDataset['DRUG']))
                    try:
                        Flist = (len(Elist)*100)/len(Dlist)
                        percentageList.append(str(str(round(Flist,3))+'%'))
                    except ZeroDivisionError:
                        Flist = "No chemicals/drugs with sufficient minimum support for this cluster"
                        percentageList.append(Flist)
                    chemicalList.append(Dlist)
                    cordUID = list(set(cordUID))
                    corduidList.append(cordUID)
                    chemCountlist.append(len(Dlist))
                    uidCountlist.append(len(cordUID))
                    includeList.append(Elist)
                    inCountlist.append(len(Elist))
                    # result = (len(intersection(listChem, list(dbDataset['DRUG'])))*100)/len(list(dbDataset['DRUG']))
                    # percentageList.append(str(str(round(result,2))+'%'))
                df_result = pd.DataFrame(
                    {
                        'Cluster Number': clusterList,
                        'List of Disease': diseaseList,
                        'List of Chemical': chemicalList,
                        'List of CORD UID': corduidList,
                        'Chemical Count': chemCountlist,
                        'CORD UID Count': uidCountlist,
                        'List of Chemicals in Drugbank': includeList,
                        'Chemicals in Drugbank Count': inCountlist,
                        'Persentage': percentageList
                    }
                )
                df_result = df_result.sort_values('CORD UID Count')
                df_result.to_csv(f"./Agglomerative/Agglomerative-Clustering-Result(affinity='{clustarg[0]}' - linkage='{clustarg[1]}').csv", index=False)
                print(f"Agglomerative-Clustering-Result(affinity='{clustarg[0]}' - linkage='{clustarg[1]}').csv saved successfully!")
                df_disease.to_csv(f"./Agglomerative/Agglomerative-Clustering-Corpus(affinity='{clustarg[0]}' - linkage='{clustarg[1]}').csv", index=False)
                print(f"Agglomerative-Clustering-Corpus(affinity='{clustarg[0]}' - linkage='{clustarg[1]}').csv saved successfully!\n")
            
            elif clustarg[1] == 'ward' and clustarg[0] == 'euclidean':
                print(f"Performing Agglomerative Clustering for affinity='{clustarg[0]}', linkage='{clustarg[1]}'")
                df_disease = pd.DataFrame(mainDataset['Disease'])
                df_disease = df_disease.drop_duplicates(subset=['Disease'])
                df_disease.reset_index(drop=True, inplace=True)
                model = AgglomerativeClustering(n_clusters=k, affinity=clustarg[0], linkage=clustarg[1])
                model.fit(data)
                df_disease['cluster'] = model.labels_

                clusterList = []
                diseaseList = []
                chemicalList = []
                corduidList = []
                chemCountlist = []
                uidCountlist = []
                includeList = []
                inCountlist = []
                percentageList = []

                for clusternumber in range(k):
                    Alist = []
                    Blist = []
                    Clist = []
                    Dlist = []
                    Elist = []
                    Flist = []
                    clusterList.append(clusternumber)
                    listDis = list(df_disease['Disease'][df_disease['cluster'] == clusternumber])
                    diseaseList.append(listDis)
                    listChem = []
                    cordUID = []
                    for disease in listDis:
                        for chem in list(mainDataset['Chemical'][mainDataset['Disease'] == disease].values):
                            listChem.append(chem)
                            # longList = list(mainDataset['Cord_UID_List'][mainDataset['Relations'] == str(disease+' --> '+chem)].values)[0].split("'")
                            # for uid in longList:
                            #     if len(uid) > 2:
                            #         cordUID.append(uid)
                    listChem = list(set(listChem))
                    Alist = listChem
                    for filtchem in Alist:
                        if filtchem not in list(rmvDataset['compounds'].values):
                            Clist.append(filtchem)
                    for dis in listDis:
                        for supchem in list(mainDataset['Chemical'][mainDataset['Disease'] == dis].values):
                            if supchem in Clist:
                                if mainDataset['Count'][mainDataset['Relations'] == str(dis+' --> '+supchem)].values >= 3:
                                    Dlist.append(supchem)
                                    longList = list(mainDataset['Cord_UID_List'][mainDataset['Relations'] == str(dis+' --> '+supchem)].values)[0].split("'")
                                    for uid in longList:
                                        if len(uid) > 2:
                                            cordUID.append(uid)
                    Elist = intersection(Dlist, list(dbDataset['DRUG']))
                    try:
                        Flist = (len(Elist)*100)/len(Dlist)
                        percentageList.append(str(str(round(Flist,3))+'%'))
                    except ZeroDivisionError:
                        Flist = "No chemicals/drugs with sufficient minimum support for this cluster"
                        percentageList.append(Flist)
                    chemicalList.append(Dlist)
                    cordUID = list(set(cordUID))
                    corduidList.append(cordUID)
                    chemCountlist.append(len(Dlist))
                    uidCountlist.append(len(cordUID))
                    includeList.append(Elist)
                    inCountlist.append(len(Elist))
                    # result = (len(intersection(listChem, list(dbDataset['DRUG'])))*100)/len(list(dbDataset['DRUG']))
                    # percentageList.append(str(str(round(result,2))+'%'))
                df_result = pd.DataFrame(
                    {
                        'Cluster Number': clusterList,
                        'List of Disease': diseaseList,
                        'List of Chemical': chemicalList,
                        'List of CORD UID': corduidList,
                        'Chemical Count': chemCountlist,
                        'CORD UID Count': uidCountlist,
                        'List of Chemicals in Drugbank': includeList,
                        'Chemicals in Drugbank Count': inCountlist,
                        'Persentage': percentageList
                    }
                )
                df_result = df_result.sort_values('CORD UID Count')
                df_result.to_csv(f"./Agglomerative/Agglomerative-Clustering-Result(affinity='{clustarg[0]}' - linkage='{clustarg[1]}').csv", index=False)
                print(f"Agglomerative-Clustering-Result(affinity='{clustarg[0]}' - linkage='{clustarg[1]}').csv saved successfully!")
                df_disease.to_csv(f"./Agglomerative/Agglomerative-Clustering-Corpus(affinity='{clustarg[0]}' - linkage='{clustarg[1]}').csv", index=False)
                print(f"Agglomerative-Clustering-Corpus(affinity='{clustarg[0]}' - linkage='{clustarg[1]}').csv saved successfully!\n")

def dbscanClusteringandEvaluations(
    data,
    rmvDataset: DataFrame,
    dbDataset: DataFrame,
    mainDataset: DataFrame,
    metric: list,
    algorithm: list,
    filename: str
    ):
    print("======================================================")
    print("========PERFORMING DBSCAN CLUSTERING ALGORITHM========")
    print("======================================================\n")
    noClust = []
    silCo = []
    CalHar = []
    DavBoul = []
    metricusedList = []

    for clustarg in list(itertools.product(metric,algorithm)):
        df_disease = pd.DataFrame(mainDataset['Disease'])
        df_disease = df_disease.drop_duplicates(subset=['Disease'])
        df_disease.reset_index(drop=True, inplace=True)
        df_loop = df_disease

        model = DBSCAN(metric=clustarg[0], algorithm=clustarg[1])
        model.fit(data)
        df_disease['cluster'] = model.labels_
        labels = model.fit_predict(data)
        # df_loop[f"DBSCAN Cluster (metric='{clustarg[0]}' - algorithm='{clustarg[1]}')"]

        clusterList = []
        diseaseList = []
        chemicalList = []
        corduidList = []
        chemCountlist = []
        uidCountlist = []
        includeList = []
        inCountlist = []
        percentageList = []
        print(f"Performing DBSCAN Clustering for metric='{clustarg[0]}', algorithm='{clustarg[1]}'")
        for clusternumber in df_disease['cluster'].unique():
            Alist = []
            Blist = []
            Clist = []
            Dlist = []
            Elist = []
            Flist = []
            clusterList.append(clusternumber)
            listDis = list(df_disease['Disease'][df_disease['cluster'] == clusternumber])
            diseaseList.append(listDis)
            listChem = []
            cordUID = []
            for disease in listDis:
                for chem in list(mainDataset['Chemical'][mainDataset['Disease'] == disease].values):
                    listChem.append(chem)
                    # longList = list(mainDataset['Cord_UID_List'][mainDataset['Relations'] == str(disease+' --> '+chem)].values)[0].split("'")
                    # for uid in longList:
                    #     if len(uid) > 2:
                    #         cordUID.append(uid)
            listChem = list(set(listChem))
            Alist = listChem
            for filtchem in Alist:
                if filtchem not in list(rmvDataset['compounds'].values):
                    Clist.append(filtchem)
            for dis in listDis:
                for supchem in list(mainDataset['Chemical'][mainDataset['Disease'] == dis].values):
                    if supchem in Clist:
                        if mainDataset['Count'][mainDataset['Relations'] == str(dis+' --> '+supchem)].values >= 3:
                            Dlist.append(supchem)
                            longList = list(mainDataset['Cord_UID_List'][mainDataset['Relations'] == str(dis+' --> '+supchem)].values)[0].split("'")
                            for uid in longList:
                                if len(uid) > 2:
                                    cordUID.append(uid)
            Elist = intersection(Dlist, list(dbDataset['DRUG']))
            try:
                Flist = (len(Elist)*100)/len(Dlist)
                percentageList.append(str(str(round(Flist,3))+'%'))
            except ZeroDivisionError:
                Flist = "No chemicals/drugs with sufficient minimum support for this cluster"
                percentageList.append(Flist)
            chemicalList.append(Dlist)
            cordUID = list(set(cordUID))
            corduidList.append(cordUID)
            chemCountlist.append(len(Dlist))
            uidCountlist.append(len(cordUID))
            includeList.append(Elist)
            inCountlist.append(len(Elist))
            # result = (len(intersection(listChem, list(dbDataset['DRUG'])))*100)/len(list(dbDataset['DRUG']))
            # percentageList.append(str(str(round(result,2))+'%'))

        df_result = pd.DataFrame(
            {
                'Cluster Number': clusterList,
                'List of Disease': diseaseList,
                'List of Chemical': chemicalList,
                'List of CORD UID': corduidList,
                'Chemical Count': chemCountlist,
                'CORD UID Count': uidCountlist,
                'List of Chemicals in Drugbank': includeList,
                'Chemicals in Drugbank Count': inCountlist,
                'Persentage': percentageList
            }
        )
        
        df_result = df_result.sort_values('CORD UID Count')
        df_result.to_csv(f"./DBSCAN/DBSCAN-Clustering-Result(metric='{clustarg[0]}' - algorithm='{clustarg[1]}').csv", index=False)
        print(f"DBSCAN-Clustering-Result(metric='{clustarg[0]}' - algorithm='{clustarg[1]}').csv saved successfully!")
        df_disease.to_csv(f"./DBSCAN/DBSCAN-Clustering-Corpus(metric='{clustarg[0]}' - algorithm='{clustarg[1]}').csv", index=False)
        print(f"DBSCAN-Clustering-Corpus(metric='{clustarg[0]}' - algorithm='{clustarg[1]}').csv saved successfully!\n")

        metricusedList.append(str(clustarg[0]+' - '+clustarg[1]))
        noClust.append(len(list(df_disease['cluster'].unique())))
        silCo.append(round(silhouette_score(data, labels), 7))
        CalHar.append(round(calinski_harabasz_score(data, labels), 7))
        DavBoul.append(round(davies_bouldin_score(data, labels), 7))
    
    df_dbscan_evaluation = pd.DataFrame(
        {
            'Metric': metricusedList,
            'Number of Generated Clusters': noClust,
            'Silhouette Score': silCo,
            'Calinski-Harabasz Score': CalHar,
            'Davies-Bouldin Score': DavBoul
        }
    )
    print(f"saving {filename}.csv...")
    df_dbscan_evaluation.to_csv(f"./DBSCAN/{filename}.csv", index=False)
    print(f"{filename}.csv saved successfully!\n")

df = pd.read_csv("./relations-counts-PubTator-377166-abstract.csv")
df_drugbank = pd.read_csv("./combined_cleaned_drugbank.csv")
df_rmv_cmp = pd.read_csv("./removed_compounds.csv")

counter = 0
disease = []
chemical = []
for relation in df['Relations']:
    disease.append(relation.split(' --> ')[0])
    chemical.append(relation.split(' --> ')[1])
df['Disease'] = disease
df['Chemical'] = chemical
df = df[["Relations", "Disease", "Chemical", "Count", "Cord_UID_List"]]
df_disease = pd.DataFrame(df['Disease'])
df_disease = df_disease.drop_duplicates(subset=['Disease'])
df_disease.reset_index(drop=True, inplace=True)

terms = list(df_disease['Disease'].values)
documents = df_disease['Disease'].values.astype("U")
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(documents)

print("\nPerforming Elbow Method for K-Means Clustering...")
kmin, kmax = 2, 50
elbow_instance = elbow(features.toarray(), kmin, kmax)  # type: ignore
elbow_instance.process()
amount_clusters = elbow_instance.get_amount()
print(f"{amount_clusters} clusters acquired\n")

kMeansClusteringEvaluation(
    nClusters= amount_clusters,
    data= features.toarray(),
    params= ["elkan", "auto", "full"],
    filename="KMeans-Clustering-Evaluation-Score"
)

kMeansClustering(
    rmvDataset= df_rmv_cmp,
    dbDataset= df_drugbank,
    mainDataset= df,
    nClusters= amount_clusters,
    data= features.toarray(),
    params= ["elkan", "auto", "full"]
)

methods = ['single', 'complete', 'average']
# distances = ['euclidean', 'minkowski', 'cosine']
affinity = ['euclidean', 'l1', 'l2', 'manhattan']
linkages = ['ward', 'complete', 'average', 'single']

agglClusteringEvaluation(
    data= features.toarray(),
    method= methods,
    distance= affinity,
    linkage= linkages,
    filename="Agglomerative-Clustering-Evaluation-Score"
)

agglClustering(
    data= features.toarray(),
    rmvDataset= df_rmv_cmp,
    dbDataset= df_drugbank,
    mainDataset= df,
    distance= affinity,
    linkage= linkages
)

dbscanClusteringandEvaluations(
    data= features.toarray(),
    rmvDataset= df_rmv_cmp,
    dbDataset= df_drugbank,
    mainDataset= df,
    metric= ['euclidean', 'l1', 'l2', 'manhattan'],
    algorithm= ['auto', 'ball_tree', 'kd_tree', 'brute'],
    filename= "DBSCAN-Clustering-Evaluation-Score"
)