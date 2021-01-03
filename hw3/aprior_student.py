from collections import defaultdict
from itertools import combinations
from sys import stdout

class cached_property(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        if obj is None: return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value

class Apriori_student:

    def __init__(self, transaction_list, minsup):
        self.transaction_list = transaction_list
        self.transaction_list_full_length = len(transaction_list)
        self.minsup = minsup

        self.frequent_itemset = dict()
        self.frequent_itemset_support = defaultdict(float)
        self.transaction_list = list([frozenset(transaction) \
            for transaction in transaction_list])

        self.rule = []

    def set_minsup(self, minsup):
        self.minsup = minsup

    @cached_property
    def items(self):
        items = set()
        for transaction in self.transaction_list:
            for item in transaction:
                items.add(item)
        return items

    def filter_with_minsup(self, itemsets):
        local_counter = defaultdict(int)
        for itemset in itemsets:
            for transaction in self.transaction_list:
                if itemset.issubset(transaction):
                    local_counter[itemset] += 1
        result = set()
        for itemset, count in local_counter.items():
            support = float(count) / self.transaction_list_full_length
            if support >= self.minsup:
                result.add(itemset)
                self.frequent_itemset_support[itemset] = support
        return result

    def freq_item(self):
        def _apriori_gen(itemset, length):
            return set([x.union(y) for x in itemset for y in itemset \
                if len(x.union(y)) == length])

        k = 1
        current_itemset = set()
        for item in self.items: current_itemset.add(frozenset([item]))
        self.frequent_itemset[k] = self.filter_with_minsup(current_itemset)
        while True:
            k += 1
            current_itemset = _apriori_gen(current_itemset, k)
            current_itemset = self.filter_with_minsup(current_itemset)
            if current_itemset != set([]):
                self.frequent_itemset[k] = current_itemset
            else:
                break
        return self.frequent_itemset

    def run(self):
        self.freq_item()
    
