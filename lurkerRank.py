
__author__ = "DIMES, University of Calabria, 2015"
import csv
import argparse
from operator import itemgetter
import numpy as np
import itertools as it
from scipy.sparse import dok_matrix
from scipy.sparse.sparsetools import csr_scale_columns
import os
import os.path as osp
import glob
from os.path import basename
from os.path import isfile
import networkx as nx




# Field separator character in the input graph file. Values on each line of the file are separated by this character
SEPARATOR = " "
# (Over)size of the node set 
DIM_MAX = 2000000
# Damping factor in LurkerRank
ALPHA = 0.85
# Maximum number of iterations 
MAX_ITERS = 150
# Convergence threshold
CONV_THRESHOLD = 1e-8



"""
Parses an input digraph file, in NCOL format <source target weight>. 
The edge meaning is that "target" node is influenced by (e.g., follows, likes) "source" node.
If "weight" is not specified, edge weights are assumed to be 1.
:param file_in: the name of the file which the data are to be read from.
:param need_mapping: a logical value indicating whether nodes are numbered using a non-progressive order.
:param header: a logical value indicating whether the file contains the names of the variables as its first line.
:return: the transpose of the adjacency matrix in CSR format.
"""
def parser_file(file_in, need_mapping, header=False):
    mapping = {} # dict for the mapping of node ids
    last_id = 0  # largest id of node 

    with open(file_in) as in_file:
        reader = csv.reader(in_file, delimiter=SEPARATOR)

        iter_reader = iter(reader)
        if header:
            # Skip header 
            next(iter_reader)

        dim = DIM_MAX
        P = dok_matrix((dim, dim))

        dim = 0
        for row in iter_reader:
            if row:
                source, target = map(int, row[:2])

                if need_mapping:
                    if source not in mapping:
                        mapping[source] = last_id
                        last_id += 1

                    if target not in mapping:
                        mapping[target] = last_id
                        last_id += 1

                    source = mapping[source]
                    target = mapping[target]
                dim = max(dim, source, target)

                if len(row) > 2:    # true if weights are available 
                    weight = float(row[2])
                    P[target, source] = weight
                else:
                    P[target, source] = 1.

        # Conversion in CSR format
        dim += 1
        P = P.tocsr()[:dim, :dim]
    return P, mapping



"""
Parses an input DiGraph object of networkx. 
If "weight" is not specified, edge weights are assumed to be 1.
:param G: the name of the DiGraph object.
:param need_mapping: a logical value indicating whether nodes are numbered using a non-progressive order.
:return: the transpose of the adjacency matrix in CSR format.
"""
def graph_to_matrix(G, need_mapping):
    mapping = {} # dict for the mapping of node ids
    last_id = 0  # largest id of node
    
    dim = DIM_MAX
    P = dok_matrix((dim, dim))
    
    dim = 0
    for edge in G.edges(data=True):
            source,target,w = edge
            if need_mapping:
                if source not in mapping:
                    mapping[source] = last_id
                    last_id += 1

                if target not in mapping:
                    mapping[target] = last_id
                    last_id += 1

                source = mapping[source]
                target = mapping[target]
            dim = max(dim, source, target)
            if len(w.keys())> 0:  # true if weights are available 
                for k in w.keys():  # if the graph is weighted then the list should contain a single attribute (e.g., 'influence' or 'weight')
                    weight = float(w[k])
                    P[target, source] = weight
                    break
            else:
                P[target, source] = 1.

    # Conversion in CSR format
    dim += 1
    P = P.tocsr()[:dim, :dim]
    return P, mapping



"""
Computes the in-degree vector and the out-degree vector.
:param P: the transpose of the adjacency matrix.
:return: I = the in-degree vector, O = the out-degree vector
"""
def compute_vector(P):
    I = P.sum(axis=1).A.ravel() + 1
    O = P.sum(axis=0).A.ravel() + 1
    return I, O



"""
All LurkerRank functions require in input: 
P: transpose of the adjacency matrix
I: in-degree vector
O: out-degree vector
Each of the functions terminates after MAX_ITERS iterations or when the error is not greater than CONV_THRESHOLD, 
and returns a unit-norm vector storing the LurkerRank solution. 
"""
def compute_LRin(P, I, O):
    n = len(I)
    r = np.ones(n) / n
    ratio = O / I
    const = (1 - ALPHA) / n
    for _ in range(MAX_ITERS):
        r_pre = r
        s = P.dot(ratio * r_pre) / O
        r = ALPHA * s + const
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r / r.sum()


def compute_LRin_out(P, I, O):
    n = len(I)
    r = np.ones(n) / n
    ratio = O / I
    const = (1 - ALPHA) / n

    sum_in = P.T.sign().dot(I)
    sum_in[sum_in == 0] = 1
    I_norm = (I / sum_in)
    for _ in range(MAX_ITERS):
        r_pre = r
        lr_in = P.dot(r_pre * ratio) / O
        lr_out = 1 + I_norm * P.T.dot(r_pre / ratio)

        r = ALPHA * lr_in * lr_out + const
        r = r / r.sum()
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r


def compute_LRout(P, I, O):
    n = len(I)
    r = np.ones(n) / n
    const = (1 - ALPHA) / n
    sum_in = P.T.sign().dot(I)
    sum_in[sum_in == 0] = 1
    ratio = I / O
    I_norm = (I / sum_in)
    for _ in range(MAX_ITERS):
        r_pre = r
        r = ALPHA * I_norm * P.T.dot(r_pre * ratio) + const
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r / r.sum()


def compute_alpha_LRin(P, I, O):
    n = len(I)
    r = np.ones(n) / n
    ratio = O / I
    for _ in range(MAX_ITERS):
        r_pre = r
        s = P.dot(ratio * r_pre) / O
        r = ALPHA * s + 1
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r / r.sum()


def compute_alpha_LRout(P, I, O):
    n = len(I)
    r = np.ones(n) / n
    sum_in = P.T.sign().dot(I)
    sum_in[sum_in == 0] = 1
    ratio = I / O
    I_norm = (I / sum_in)
    for _ in range(MAX_ITERS):
        r_pre = r
        r = ALPHA * I_norm * P.T.dot(r_pre * ratio) + 1
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r / r.sum()


def compute_alpha_LRin_out(P, I, O):
    n = len(I)
    r = np.ones(n) / n
    ratio = O / I

    sum_in = P.T.sign().dot(I)
    sum_in[sum_in == 0] = 1
    I_norm = (I / sum_in)
    for _ in range(MAX_ITERS):
        r_pre = r
        lr_in = P.dot(r_pre * ratio) / O
        lr_out = 1 + I_norm * P.T.dot(r_pre / ratio)

        r = ALPHA * lr_in * lr_out + 1
        r = r / r.sum()
        if np.allclose(r, r_pre, atol=CONV_THRESHOLD, rtol=0):
            break
    return r


LR_METHODS = {'LRin': compute_LRin,
             'LRinout': compute_LRin_out,
             'LRout': compute_LRout,
             'acLRin': compute_alpha_LRin,
             'acLRinout': compute_alpha_LRin_out,
             'acLRout': compute_alpha_LRout
             }




def print_result(res, mapping, out_file_path):
    nodes = np.arange(0, len(res))
    nodes = np.lexsort((nodes, -res))
    res = res[nodes]

    flat_mapp = []
    if mapping:
        while mapping:
            flat_mapp.append(mapping.popitem())
        flat_mapp.sort(key=lambda t: t[1])

    with open(out_file_path, "w") as out_file:
        for user_id, val in zip(nodes, res):
            if flat_mapp:
                user_id = flat_mapp[user_id][0]
            print(user_id, val, file=out_file, sep=";")



def lr_dict(res, mapping):
    lr_dict = {}
    nodes = np.arange(0, len(res))
    nodes = np.lexsort((nodes, -res))
    res = res[nodes]

    flat_mapp = []
    if mapping:
        while mapping:
            flat_mapp.append(mapping.popitem())
        flat_mapp.sort(key=lambda t: t[1])
    for user_id, val in zip(nodes, res):
            if flat_mapp:
                user_id = flat_mapp[user_id][0]
            lr_dict[int(user_id)] = float(val)
    return lr_dict


"""
Invokes a particular LurkerRank function. 
:param func: the name of a LurkerRank function selected from the enumeration LR_METHODS
:param file_in: the name of the file which the data are to be read from.
:param file_out: the name of the file where the results are to be printed out.
:param need_mapping: a logical value indicating whether nodes are numbered using a non-progressive order.
:param return_dict: if True, the function returns a dict <id,score>, otherwise results are printed out to file_out.
:param input_graph: True if file_in corresponds to a DiGraph object of networkx.
"""
def computeLR(func, file_in, file_out, need_mapping=True, return_dict=False, input_graph=False):
    if input_graph==True:
        P, mapping = graph_to_matrix(file_in, need_mapping)
    else:
        P, mapping = parser_file(file_in, need_mapping)
    # print("Graph matrix has been created.\n")
    I, O = compute_vector(P)
    # print("Degree vectors have been created.\n") 
    res = func(P, I, O)
    if return_dict==False:
        print_result(res, mapping, file_out)
    else:
        return lr_dict(res,mapping)



"""
:param dir: the path of the input directory.
:param out: the path of the output directory.
:param need_mapping: a logical value indicating whether nodes are numbered using a non-progressive order.
:param method: the name of the LurkerRank function to run.
"""
def compute_on_dir(dir,out,method,need_mapping=True):
    graphs = glob.glob(dir+"*.ncol")
    for sg_path in graphs: 
        in_file = sg_path
        func = LR_METHODS[method]  
        filename = basename(sg_path).split('.')[0]
        out_file = out+filename+"_"+method+".txt"
        if not isfile(out_file):
            folder = osp.split(out_file)[0]
            try:
                os.makedirs(folder)
            except OSError:
               pass
        computeLR(func, in_file, out_file, need_mapping)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'inputDir',  help='path of the folder containing the input graph file(s)')
    parser.add_argument(
        'outputDir',  help='path of the folder containing the output file(s) storing LurkerRank solution(s)')
    parser.add_argument('-lr', required=False, choices=['LRin', 'LRout', 'LRinout','acLRin', 'acLRout', 'acLRinout'], default='LRin', help='LurkerRank method (default: LRin)')
    
    args = parser.parse_args()

    compute_on_dir(args.inputDir,args.outputDir,args.lr)
