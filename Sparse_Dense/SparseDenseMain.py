from Sparse_Dense.sparse_flow import SparseOpticalFlow
from Sparse_Dense.dense_flow import DenseOpticalFlow

#https://nanonets.com/blog/optical-flow/
sparse_or_dense = True

def doSparse():
    sparse = SparseOpticalFlow
    sparse.Start(sparse)

def doDense():
    dense = DenseOpticalFlow
    dense.Start(dense)


if sparse_or_dense:
    doSparse()
else:
    doDense()

