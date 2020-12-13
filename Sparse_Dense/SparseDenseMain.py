from Sparse_Dense.sparse_flow import SparseOpticalFlow
from Sparse_Dense.dense_flow import DenseOpticalFlow
from Sparse_Dense.sparse_flow_v2 import SparseOpticalFlowMod

#https://nanonets.com/blog/optical-flow/
sparse_or_dense = True
sparse_modified = True

def doSparse():
    if sparse_modified:
        sparseMod = SparseOpticalFlowMod
        sparseMod.Start(sparseMod)
    else:
        sparse = SparseOpticalFlow
        sparse.Start(sparse)

def doDense():
    dense = DenseOpticalFlow
    dense.Start(dense)


if sparse_or_dense:
    doSparse()
else:
    doDense()

