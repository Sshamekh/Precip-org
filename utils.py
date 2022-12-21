import numpy as np
def bin_threshold(data1,data2,nbins):
    minpw = data1.min()
    maxpw = data1.max()
    binsize = ((maxpw - minpw)/nbins)
    print (minpw,maxpw,binsize)
    bined_pw = []
    bined_precip = []
    std1 = []
    std2 = []
    for i in range(nbins+1):
        th1 = minpw+i*binsize 
        th2 = minpw+(i+1)*binsize 
        indecies = np.where((data1>=th1) & (data1<th2))[0]
        if len(indecies)>10:
            bined_pw.append(np.mean(data1[indecies]))
            bined_precip.append(np.mean(data2[indecies]))
            std1.append(np.std(data1[indecies]))
            std2.append(np.std(data2[indecies]))
            
    return bined_pw,bined_precip,std1,std2

def gini(array):
    """Calculate the Gini coefficient of a numpy array.
        array: a 3D field with nsamples, nx,ny
        gini index is computed for plane xy
    """
    samples , nx,ny = array.shape 
    array = array.reshape(samples,nx*ny)
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array,axis = 1)
    # Index per array element:
    index = np.arange(1,array.shape[1]+1).reshape(1,-1)
    # Number of array elements:
    n = array.shape[1]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array,axis = 1 )) / (n * np.sum(array,axis = 1)))

