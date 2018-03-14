import numpy as np
def relativeEntropy(baseLabel, relativeLabel):
    """
    Calculate the relative entropy of the two given input label list.
    """
    clusterDict = dict()
    for b, r in zip(baseLabel, relativeLabel):
        rList = clusterDict.get(r)
        if rList is None:
            rList = list()
            clusterDict[r] = rList
        rList.append(b)
    
    count = len(relativeLabel)
    e = 0.0
    for _, baseLabelList in clusterDict.items():
        e += entropy(baseLabelList) * len(baseLabelList) / count
    return e

def entropy(labels):
    """
    Calculate the labels of  the labels.
    """
#     print labels
    A = np.asarray(labels, dtype=np.int)
    A = A.flatten()
    counts = np.bincount(A)  # needs small, non-negative ints
    counts = counts[counts > 0]
    if len(counts) == 1:
        return 0.  # avoid returning -0.0 to prevent weird doctests
    probs = counts / float(A.size)
    return -np.sum(probs * np.log2(probs))

if __name__ == '__main__':
    baseLabel = [1, 1, 1, 2, 2, 2]
    relativeLabel = [1, 2, 2, 2, 2, 2]
    print relativeEntropy(baseLabel, relativeLabel)
