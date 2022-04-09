import sys, os
from os.path import dirname, basename
import numpy as np
def compute_fr(predictions_file_name):
    l = {}
    method_name = None
    perturbation_name = None
    with open(predictions_file_name) as f:
        for x in f:
            if x.startswith('#'): continue
            x = [x.strip() for x in x.split(',')]
            if len(x)<2:
                x = [ "ext", x[0] ]
            if len(x)==2:
                # Save accumulated result for previous method/perturbation if any
                if perturbation_name is not None and method_name is not None and K>0:
                    result /= K
                    l[method_name][perturbation_name] = result
                # Start accumulation for next method/perturbation
                result = 0
                K = 0 # num of sample sequences
                method_name = basename(dirname(x[0]))
                perturbation_name = basename(x[1])
                if not method_name in l:
                    l[method_name] = {}
            else:
                # Accumulate data
                x = [int(a) for a in x]
                N = len(x) # lenght of a sequence
                assert( 30 == N )
                if '_noise' in perturbation_name:
                    x1 = np.array(x[1:], np.uint8)
                    x2 = np.array(x[0], np.uint8)
                else:
                    x1 = np.array(x[1:], np.uint8)
                    x2 = np.array(x[:-1], np.uint8)
                result += np.sum(x1!=x2) / (N-1)
                K +=1
    result /= K
    l[method_name][perturbation_name] = result
    return l

def sort(d):
    import collections
    d = collections.OrderedDict(sorted(d.items()))
    return d

def print_text(l):
    for m,pres in l.items():
        print(m)
        for p,res in pres.items():
            print(p,res)
        print("---------")

def print_xls(l):
    l = sort(l)
    perturbs = list(l.values())[-1].keys()
    print('.', end='\t')
    # Print all method headers
    for m,pres in l.items():
        if len(pres.keys())>=len(perturbs):
            print(m, end='\t')
    print()
    # Print one perturb per row
    for p in perturbs:
        # print results for each method
        print(p.replace('rafdb.test.',''), end='\t')
        for m,pres in l.items():
            if len(pres.keys())>=len(perturbs):
                print(pres[p], end='\t')
        print()

if "__main__"==__name__:
    l = compute_fr(sys.argv[1])
    if len(sys.argv)>2 and sys.argv[2][0]=='t':
        print_text(l)
    else:
        print_xls(l)
