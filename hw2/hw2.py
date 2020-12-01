import re
import os
import warnings
import numpy
import sys
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import jaccard_score
from sklearn.utils.multiclass import type_of_target
from sklearn import manifold
from scipy.spatial import distance
from scipy import sparse
import matplotlib.pyplot as plt

X_norm = numpy.matrix('1 2; 3 4')
color_plate = {
-1 : '#000020',0 : 'blue',1: 'green',2: 'red',3: 'cyan',4: 'magenta',5: 'yellow', 6: 'black',7: '#202060',
8: '#206020',9: '#602020',10: '#206060',11: '#606060',12: '#808080',13: '#806040', 14: '#A02020', 15: '#A00000',16: '#B02020', 17: '#F0F020',
18: '#8000F0', 19: '#F0F020', 20: '#802000'}
def Drawer(label,filename):
   
    for i in range(X_norm.shape[0]):
        if label[i] > 20:
            color = label[i] % 21
        else:
            color = label[i]
        plt.plot(X_norm[i,0],X_norm[i,1],marker='o',color=color_plate[color])
    plt.savefig(filename)   
    plt.clf()

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
                    self.user_product[userId].add(productId)
                else:
                    product_list = set()
                    product_list.add(productId)
                    self.user_product[userId] = product_list
                self.userId_list = sorted( self.userId_set )
                self.productId_list = sorted( self.productId_set )
        
    def process_each_userId(self):
        index = []
        column = []
        value = []
        user_index = 0
        for user in self.userId_list:
            for product in self.user_product[user]:
                index.append( user_index )
                column.append( self.productId_list.index(product) )
                value.append ( 1 )
            user_index += 1
        self.sparse_matrix = sparse.csc_matrix((value,(index,column)))

    def process_kmeasn_euclidean(self):
        mat = self.sparse_matrix
        kmeans = KMeans(n_clusters=10)
        kmeans.fit(mat)
        Drawer(kmeans.labels_,"kmeans.png")
        print("Kmeans:")
        print(kmeans.inertia_)
        return kmeans.inertia_

    def return_mat(self):
        return self.sparse_matrix
    
    def read_dataframe_file(self, filename):
        self.df = pd.read_pickle(filename)

    def process_dbscan_jaccard(self):
        mat = self.sparse_matrix
        dbscan = DBSCAN()
        dbscan.fit(mat)
        row, col = self.sparse_matrix.get_shape()

        #Calculate average point position for each cluster
        cluster = {}
        clusetr_amount = {}
        for i in range(0,row) :
            if dbscan.labels_[i] != -1 :
                for j in range(0,col):
                    if  self.sparse_matrix[i,j] == 1 :
                        if dbscan.labels_[i] in cluster:
                           clusetr_amount[dbscan.labels_[i]] = clusetr_amount[dbscan.labels_[i]] +1
                           cluster[dbscan.labels_[i]][j] = 1
                        else:
                            cluster[dbscan.labels_[i]] = numpy.zeros(col)
                            cluster[dbscan.labels_[i]][j] = 1
                            clusetr_amount[dbscan.labels_[i]] = 1

        for key in cluster:
            cluster[key] = cluster[key]/clusetr_amount[key]
        
        minumum_distance = {}
        for key in cluster:
            minumum_distance[key] = 999

        #Find the nearest row by euclidean to average point position as centroid for each cluster
        array_row = numpy.zeros(col)
        centroid = {}
        for i in range(0,row) :
            if dbscan.labels_[i] != -1 :
                for j in range(0,col):
                    if self.sparse_matrix[i,j] == 1 :
                        array_row[j] = 1
                eu_dist = distance.euclidean(array_row, cluster[key])
                if eu_dist < minumum_distance[dbscan.labels_[i]]:
                    minumum_distance[dbscan.labels_[i]] = eu_dist
                    centroid[dbscan.labels_[i]] = i
            array_row = numpy.zeros(col)
        
        centroid_matrix = numpy.zeros((len(centroid), col))
        for i in range(0,len(centroid_matrix)):
            for j in range(0,col):
                centroid_matrix[i][j] = self.sparse_matrix[centroid[i],j]
        
        #Find overall jaccard score by centroid
        array_row = numpy.zeros(col)
        overall_distance = 0 
        for i in range(0,row) :
            if dbscan.labels_[i] != -1 :
                for j in range(0,col):
                    if self.sparse_matrix[i,j] == 1 :
                        array_row[j] = 1
                ja_distance = jaccard_score( centroid_matrix[dbscan.labels_[i]], array_row)
                overall_distance = overall_distance + ja_distance
            array_row = numpy.zeros(col)
        return overall_distance

class Kmeans_Jaccard():
    def __init__(self, cluster_num, dimension):
        self.dimension = dimension
        self.cluster_num = cluster_num
        self.cluster_centroids = numpy.random.rand(cluster_num,dimension)
        self.cluster_centroids = self.cluster_centroids.astype(uint8)
        self.label = {}

    def dis(self, x, y):
            return jaccard_score( x, y)


    def cluster(self, matrix):
        mat = matrix.values
        index = 0
        label_ = {}
        for row in mat:
            ja_distance = 2 #maximum value is 1 
            nearest_cluster = -1
            i = 0
            for centroids in self.cluster_centroids :
                jaccard = self.dis( numpy.array(centroids).astype(uint8), numpy.array(row).astype(uint8))
                if jaccard < ja_distance:
                    ja_distance = jaccard
                    nearest_cluster = i
                i += 1
            label_[index] = nearest_cluster
            index += 1
        if label_ == self.label:
            return True
        else:
            self.label = label_.copy()
            return False

    def recal_centroids(self,matrix):
        mat = matrix.values
        cluster_amount = numpy.zeros(self.cluster_num)
        sum_matrix = numpy.zeros((self.cluster_num, self.dimension))
        for i in range(0, len(mat)) :
            sum_matrix[self.label[i]] = numpy.array(mat[i]) + numpy.array(sum_matrix[self.label[i]])
            cluster_amount[self.label[i]] = cluster_amount[self.label[i]] + 1
        for j in range(0, len(sum_matrix)):
            for k in range(0,len(sum_matrix[j])):
                if sum_matrix[j][k] > 0 and cluster_amount[j] > 0:
                    sum_matrix[j][k] =  sum_matrix[j][k] / cluster_amount[j]
                else:
                    sum_matrix[j][k] = 0
        new_centroids = numpy.zeros((self.cluster_num, self.dimension))
        minumum_distance = []
        for j in range(0,len(sum_matrix)):
            minumum_distance.append(999)
            
        i = 0
        for row in mat :
            eu_dist = distance.euclidean(sum_matrix[self.label[i]], row)
            if eu_dist < minumum_distance[self.label[i]] :
                new_centroids[self.label[i]] = numpy.copy(row)
                minumum_distance[self.label[i]] = eu_dist
            i += 1 
        self.cluster_centroids = numpy.copy(new_centroids)
 
    def process_kmeans(self,matrix):
        counter = 0
        while True:
            if True ==  self.cluster(matrix):
                break
            self.recal_centroids(matrix)
            counter += 1
            if counter >= 300:
                break
        print(counter)


        
if __name__ == "__main__":
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    numpy.set_printoptions(threshold=sys.maxsize)
    hw = HW2("test.txt")
    #X_tsne = manifold.TSNE(n_components=2).fit_transform(hw.return_df().values)
    #x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    #X_norm = ( X_tsne - x_min ) / (x_max - x_min)
    #for i in range(X_norm.shape[0]):
    #    plt.plot(X_norm[i,0],X_norm[i,1],'ro')
    #plt.show()
    hw.parse()
    hw.process_each_userId()
    hw.process_kmeasn_euclidean()
    hw.process_dbscan_jaccard()
    #hw.transform_to_dataframe()
    X_tsne = manifold.TSNE(n_components=2).fit_transform(hw.return_mat())
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = ( X_tsne - x_min ) / (x_max - x_min)
    for i in range(X_norm.shape[0]):
        plt.plot(X_norm[i,0],X_norm[i,1],marker='o',color='black')
    plt.show()  
    #hw.process_kmeasn_euclidean()
    #hw.process_dbscan_jaccard()
    #kmeans_jaccard = Kmeans_Jaccard(10, hw.return_df().values.shape[1])
    #kmeans_jaccard.process_kmeans(hw.return_df())

    #for i in range(X_norm.shape[0]):
    #    plt.plot(X_norm[i,0],X_norm[i,1],marker='o',color='black')
    #plt.savefig("orig.png")  