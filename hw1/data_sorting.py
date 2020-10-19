import re
import os
import warnings
import numpy
from fpdf import FPDF 

class HW1():
    def __init__(self, filename, ofilename, hw):
        self.fptr = open(filename, mode='r')
        self.optr = open(ofilename, 'w')
        self.userId_attr = "review/userId:"
        self.productId_attr = "product/productId:"
        self.user_product = dict()
        self.userId_set = set()
        self.productId_set = set()
        self.median = 0
        self.writer = open(hw, 'w')

    def __del__(self):
        self.fptr.close()
        self.optr.close()
        self.writer.close()
        

    def parser(self):
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
                        self.userId_set.add(userId)
                        break

                if userId in self.user_product:
                    self.user_product[userId].append(productId)
                else:
                    product_list = []
                    product_list.append(productId)
                    self.user_product[userId] = product_list

    def show_ret(self):
        max_product_num = 0
        max_product_user = ''
        product_amount_list = []
        for key, value in self.user_product.items():
            self.optr.write(key)
            product_amount_list.append(len(value))
            if len(value) > max_product_num and key != "unknown":
                max_product_num = len(value)
                max_product_user = key
            for product in value:
                self.optr.write(',')
                self.optr.write(product)
            self.optr.write('\n')
        self.writer.write(
            "How many unique users are in the Music.txt.gz dataset?\n")
        self.writer.write("Ans: " + str(len(self.userId_set) - 1)+"\n")
        self.writer.write(
            "How many unique products are in the Music.txt.gz dataset?\n")
        self.writer.write("Ans: " + str(len(self.user_product))+"\n")
        self.writer.write(
            "What is the maximum number of products bought by a user?\n")
        self.writer.write("Ans: max amount is " + str(max_product_num) +
                        " user id is " + max_product_user + "\n")
        self.median = numpy.median(product_amount_list)

    def median_sort(self):
        median_userId = []
        self.writer.write("What is the value of the median of the amount of products bought?\nFor those users"
                        "having the median number of products, sort their userIds in ascending lexicographic order.\n"
                        "Then, print the first ten userIds if there are more than ten or print all of them if not \n")
        for key, value in self.user_product.items():
            if len(value) == self.median:
                median_userId.append(key)
        median_userId = sorted(median_userId)
        self.writer.write("Ans: \n")
        self.writer.write("median: "+str(self.median) + "\n")
        self.writer.writelines("%s \n" % item for item in median_userId[0:10])
    
def to_pdf(in_file, o_file):
     
    pdf = FPDF()    
    
    pdf.add_page() 
    
    pdf.set_font("Arial", size = 12 )
    
    f = open(in_file, "r") 
    
    for x in f: 
        pdf.cell(200, 10, txt = x, ln = 1, align = 'L') 

    pdf.output(o_file)    

if __name__ == "__main__":
    hw1 = HW1("Music.txt", "result.csv", "509557006.txt")
    hw1.parser()
    hw1.show_ret()
    hw1.median_sort()
    del hw1
    to_pdf("509557006.txt","509557006.pdf")
