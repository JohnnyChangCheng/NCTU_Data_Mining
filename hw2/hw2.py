import re
import os
import warnings
import numpy
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import jaccard_score
from sklearn.utils.multiclass import type_of_target
class HW2():
    def __init__(self, filename):
        self.fptr = open(filename, mode='r')
        self.userId_attr = "review/userId:"
        self.productId_attr = "product/productId:"
        self.user_product = dict()
        self.userId_set = set()
        self.productId_set = set()
        self.median = 0

    def __del__(self):
        self.fptr.close()
        

    def parse(self):
        while True:
            buf = self.fptr.readline()
            if not buf:
                return
            if re.search(self.productId_attr, buf):
                userId = ''
                buf = buf.strip(self.productId_attr)
                buf = buf.strip(" ")
                productId = buf.strip('\n')
                self.productId_set.add(productId)

                while True:
                    buf = self.fptr.readline()
                    if not buf:
                        warnings.warn('file eof unexpected', RuntimeWarning)
                        return
                    if re.search(self.userId_attr, buf):
                        buf = buf.strip(self.userId_attr)
                        buf = buf.strip(" ")
                        userId = buf.strip('\n')
                        if userId.find("unknown") != -1 :
                            userId = "unknown"
                        self.userId_set.add(userId)
                        break

                if userId in self.user_product:
                    self.user_product[userId].append(productId)
                else:
                    product_list = []
                    product_list.append(productId)
                    self.user_product[userId] = product_list

    def transform_to_dataframe(self):
        self.df = pd.DataFrame(index=sorted(self.userId_set), columns=sorted(self.productId_set) )
        for userId in sorted(self.userId_set):
            self.process_each_userId(userId)
        self.df.to_pickle("frame.pickle")
        
    def process_each_userId(self, userId):
        for item in sorted(self.productId_set):
            if item in self.user_product[userId]:
                self.df.loc[[userId],[item]] = 1
            else:
                self.df.loc[[userId],[item]] = 0

    def process_kmeasn_euclidean(self):
        mat = self.df.values
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(mat)
        return kmeans.inertia_

    def read_dataframe_file(self, filename):
     self.df = pd.read_pickle(filename)

    def process_dbscan_jaccard(self):
        mat = self.df.values
        dbscan = DBSCAN()
        dbscan.fit(mat)
        i = 0
        cluster = {}
        centroids = {}
        clusetr_amount = {}
        for row in mat :
            if dbscan.labels_[i] != -1 :
                if dbscan.labels_[i] in cluster:
                    clusetr_amount[dbscan.labels_[i]] = clusetr_amount[dbscan.labels_[i]] +1
                    cluster[dbscan.labels_[i]] = cluster[dbscan.labels_[i]] +  numpy.array(row)
                else:
                    cluster[dbscan.labels_[i]] =  numpy.array(row)
                    clusetr_amount[dbscan.labels_[i]] = 1
            i = i + 1 
        for key in cluster:
            centroids[key] = cluster[key]/clusetr_amount[key]
        i = 0
        overall_distance = 0
        for row in mat :
            if dbscan.labels_[i] != -1 :
                point = centroids[dbscan.labels_[i]]
                distance = jaccard_score( point.astype(int), numpy.array(row).astype(int))
                overall_distance = overall_distance + distance
            i = 1 + i
        return overall_distance



if __name__ == "__main__":
    hw = HW2("test.txt")
    #hw.parse()
    #hw.transform_to_dataframe()
    hw.read_dataframe_file("frame.pickle")
    hw.process_kmeasn_euclidean()
    hw.process_dbscan_jaccard()

