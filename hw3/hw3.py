import re
import os
import warnings
import numpy
import sys
import pandas as pd
import timeit
import matplotlib.pyplot as plt
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth
from aprior_student import *

def plot_all(ap,fp,stu):
    plt.figure(1)
    x_axi = ["sup=0.0006","sup=0.00075","sup=0.0009"]
    plt.plot(x_axi, ap, 's-',color='r',label="Apriori")
    plt.plot(x_axi, fp, 'o-',color='b',label="Fpgrowth")
    plt.plot(x_axi, stu, 'v-',color='g',label="Apriori-student")
    plt.ylabel("Execution time(Seconds)")
    plt.xlabel("Support")
    plt.legend(loc = "best")
    plt.savefig("Profile_all.png")
    plt.clf()

def plot_music(ap,fp,stu):
    plt.figure(1)
    x_axi = ["sup=0.0003","sup=0.0006","sup=0.0009"]
    plt.plot(x_axi, ap, 's-',color='r',label="Apriori")
    plt.plot(x_axi, fp, 'o-',color='b',label="Fpgrowth")
    plt.plot(x_axi, stu, 'v-',color='g',label="Apriori-student")
    plt.ylabel("Execution time(Seconds)")
    plt.xlabel("Support")
    plt.legend(loc = "best")
    plt.savefig("Profile_music.png")
    plt.clf()

class HW3():
    def __init__(self, filename):
        self.fotr = open("hw3.log", mode='w')
        self.fptr = open(filename, mode='r')
        self.userId_attr = "review/userId:"
        self.productId_attr = "product/productId:"
        self.user_product = dict()
        self.userId_list = []
        self.dataset = []
 
    def parser(self):
        #Process raw data
        userId_set = set()
        while True:
            buf = self.fptr.readline()
            if not buf:
                break
            if re.search(self.productId_attr, buf):
                userId = ''
                buf = buf.strip(self.productId_attr)
                buf = buf.strip(" ")
                productId = buf.strip('\n')

                while True:
                    buf = self.fptr.readline()
                    if not buf:
                        warnings.warn('file eof unexpected', RuntimeWarning)
                        break
                    if re.search(self.userId_attr, buf):
                        buf = buf.strip(self.userId_attr)
                        buf = buf.strip(" ")
                        userId = buf.strip('\n')
                        if userId.find("unknown") != -1:
                            userId = "unknown"
                        userId_set.add(userId)
                        break

                if userId in self.user_product:
                    self.user_product[userId].add(productId)
                else:
                    product_list = set()
                    product_list.add(productId)
                    self.user_product[userId] = product_list

        self.userId_list = sorted(userId_set)
        self.fotr.write("Process meta dataset finished \n")


    def process_dataframe(self):
        #Using Sparse matrix to store the data
        for user in self.userId_list:
            newlist = list(self.user_product[user])
            self.dataset.append(newlist)
        self.user_product.clear()
        te = TransactionEncoder()
        oht_ary = te.fit(self.dataset).transform(self.dataset, sparse=True)
        sparse_df = pd.DataFrame.sparse.from_spmatrix(oht_ary, columns=te.columns_)
        sparse_df.to_pickle("Sparse.pickle")
        self.fotr.write("Dataset to dataframe finished \n")
        return sparse_df, self.dataset

    def load_dataframe(self):
        return pd.read_pickle("Sparse.pickle")

    def ariori_dataset(self, sparse_df):
        tic = timeit.default_timer()
        frequent_itemsets = apriori(sparse_df, min_support=0.0009, use_colnames=True, verbose=0,low_memory=True)
        toc=timeit.default_timer()
        self.fotr.write("The apriori with 0.0009 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
        len3_itemsets = frequent_itemsets[ frequent_itemsets['length'] >= 3 ]
        len3_itemsets = len3_itemsets.sort_values(by=['length','itemsets'],ascending=[False,True])
        self.fotr.write("The item len >= 3 is " + str(len3_itemsets.shape[0]))
        self.fotr.write("\n")
        self.fotr.flush()
        top10_row_len3_itemsets = len3_itemsets.head(10)
        top10_row_len3_itemsets.to_csv ("Top10_length.csv", index = True, header=True)
        return frequent_itemsets
    
    def rule_generater(self, itemsets):
        rule_generate = association_rules(itemsets, metric="confidence", min_threshold=0.7)
        top10_rule_generate = rule_generate.sort_values(by=['confidence'],ascending=[False]).head(10)
        top10_rule_generate.to_csv ("Top10_confidence.csv", index = True, header=True)
        self.fotr.write("Finish rule generater\n")
        return rule_generate
    
    def time_profile_music(self,sparse_df):
        apr_time = []
        fp_time = []

        tic = timeit.default_timer()
        apriori(sparse_df, min_support=0.0003, use_colnames=True, verbose=0,low_memory=True)
        toc=timeit.default_timer()
        self.fotr.write("The apriori with 0.0003 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        apr_time.append(toc-tic)

        tic = timeit.default_timer()
        apriori(sparse_df, min_support=0.0006, use_colnames=True, verbose=0,low_memory=True)
        toc=timeit.default_timer()
        self.fotr.write("The apriori with 0.0006 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        apr_time.append(toc-tic)

        tic = timeit.default_timer()
        apriori(sparse_df, min_support=0.0009, use_colnames=True, verbose=0,low_memory=True)
        toc=timeit.default_timer()
        self.fotr.write("The apriori with 0.0009 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        apr_time.append(toc-tic)

        tic = timeit.default_timer()
        fpgrowth(sparse_df, min_support=0.0003, use_colnames=True)
        toc=timeit.default_timer()
        self.fotr.write("The fpgrowth with 0.0003 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        fp_time.append(toc-tic)

        tic = timeit.default_timer()
        fpgrowth(sparse_df, min_support=0.0006, use_colnames=True)
        toc=timeit.default_timer()
        self.fotr.write("The fpgrowth with 0.0006 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        fp_time.append(toc-tic)

        tic = timeit.default_timer()
        fpgrowth(sparse_df, min_support=0.0009, use_colnames=True)
        toc=timeit.default_timer()
        self.fotr.write("The fpgrowth with 0.0009 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        fp_time.append(toc-tic)

        return apr_time, fp_time

    def time_aps_profile_music(self,dataset):
        aps_time=[]
        apriori = Apriori_student(dataset, 0.0003)

        tic = timeit.default_timer()
        apriori.run()
        toc=timeit.default_timer()
        self.fotr.write("The Apr-student with 0.0003 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        aps_time.append(toc-tic)

        tic = timeit.default_timer()
        apriori.set_minsup(0.0006)
        apriori.run()
        toc=timeit.default_timer()
        self.fotr.write("The Apr-student with 0.0006 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        aps_time.append(toc-tic)

        tic = timeit.default_timer()
        apriori.set_minsup(0.0009)
        apriori.run()
        toc=timeit.default_timer()
        self.fotr.write("The Apr-student with 0.0009 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        aps_time.append(toc-tic)

        return aps_time

    def time_profile_all(self,sparse_df):
        apr_time = []
        fp_time = []

        tic = timeit.default_timer()
        apriori(sparse_df, min_support=0.0006, use_colnames=True, verbose=0,low_memory=True)
        toc=timeit.default_timer()
        self.fotr.write("The apriori with 0.0006 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        apr_time.append(toc-tic)

        tic = timeit.default_timer()
        apriori(sparse_df, min_support=0.00075, use_colnames=True, verbose=0,low_memory=True)
        toc=timeit.default_timer()
        self.fotr.write("The apriori with 0.00075 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        apr_time.append(toc-tic)

        tic = timeit.default_timer()
        apriori(sparse_df, min_support=0.0009, use_colnames=True, verbose=0,low_memory=True)
        toc=timeit.default_timer()
        self.fotr.write("The apriori with 0.0009 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        apr_time.append(toc-tic)

        tic = timeit.default_timer()
        fpgrowth(sparse_df, min_support=0.0006, use_colnames=True)
        toc=timeit.default_timer()
        self.fotr.write("The fpgrowth with 0.0006 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        fp_time.append(toc-tic)

        tic = timeit.default_timer()
        fpgrowth(sparse_df, min_support=0.00075, use_colnames=True)
        toc=timeit.default_timer()
        self.fotr.write("The fpgrowth with 0.00075 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        fp_time.append(toc-tic)

        tic = timeit.default_timer()
        fpgrowth(sparse_df, min_support=0.0009, use_colnames=True)
        toc=timeit.default_timer()
        self.fotr.write("The fpgrowth with 0.0009 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        fp_time.append(toc-tic)

        return apr_time, fp_time

    def time_aps_profile_all(self,dataset):
        aps_time=[]
        apriori = Apriori_student(dataset, 0.0006)

        tic = timeit.default_timer()
        apriori.run()
        toc=timeit.default_timer()
        self.fotr.write("The Apr-student with 0.0006 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        aps_time.append(toc-tic)

        tic = timeit.default_timer()
        apriori.set_minsup(0.00075)
        apriori.run()
        toc=timeit.default_timer()
        self.fotr.write("The Apr-student with 0.00075 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        aps_time.append(toc-tic)

        tic = timeit.default_timer()
        apriori.set_minsup(0.0009)
        apriori.run()
        toc=timeit.default_timer()
        self.fotr.write("The Apr-student with 0.0009 elaspe " + str(toc-tic))
        self.fotr.write("\n")
        self.fotr.flush()
        aps_time.append(toc-tic)

        return aps_time

if __name__ == "__main__":
    
    mode = sys.argv[1]
    
    if mode == "Music" :
        hw = HW3("Music.txt")
        hw.parser()
        df_raw_data, dataset = hw.process_dataframe()
        df = hw.ariori_dataset(df_raw_data)
        df = hw.rule_generater(df)
        ap_time, fpg_time = hw.time_profile_music(df_raw_data)
        aps_time = hw.time_aps_profile_music(dataset)
        plot_music(ap_time, fpg_time, aps_time)

    elif mode == "All":
        hw = HW3("all.txt")
        hw.parser()
        df_raw_data, dataset = hw.process_dataframe()
        ap_time, fpg_time = hw.time_profile_all(df_raw_data)
        aps_time = hw.time_aps_profile_all(dataset)
        plot_all(ap_time, fpg_time, aps_time)
    

  