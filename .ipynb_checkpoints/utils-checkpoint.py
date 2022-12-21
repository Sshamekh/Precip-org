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


