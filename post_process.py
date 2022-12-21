import numpy as np
from sklearn.metrics import r2_score
from netCDF4 import Dataset
#from global_land_mask import globe


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
        if len(indecies)>5:
            bined_pw.append(np.mean(data1[indecies]))
            bined_precip.append(np.mean(data2[indecies]))
            std1.append(np.std(data1[indecies]))
            std2.append(np.std(data2[indecies]))
            
    return bined_pw,bined_precip,std1,std2


def Compute_R2_lonlat(data,true,model,thsh):
    
    """
    A function to compute R2 for each lat-lon and across time steps. 
    Data: input to the model with shape (samples, timewindow,lat,lon,feature)
    
    True is the output true value. Shape is (samples, timewindow,lat,lon,feature)
       Timewindow is equal to one for baseline model and two or more for  models
       that include few time steps, the prediction is always for the last time
       step of the time window.
    
    Thsh is the threshold of precipitation that we used for preprocessing the data. 
    
    We exclude all datapoint with precip smaller than threshold
    """
    
    nt,t_wind, lat, lon, features = data.shape
    R2matrix = np.zeros((lat,lon))
    R2matrix[:] = np.nan
    for yy in range(lat):
        for xx in range(lon):
            test = data[:,:,yy,xx,:]
            true_i = true[:,:,yy,xx,:]
            
            index = np.where(true_i[:,-1,0]>thsh)[0]
            if len(index)>10:
                test = (test[index,:]).reshape(-1,t_wind,features)
                true_i = true_i[index,:]
                predict = model.predict(test)
                R2matrix[yy,xx] = r2_score(predict[:,0],true_i[:,-1,0])
    return R2matrix



def lon_lat():
    ds = Dataset('/glade/scratch/sshamekh/dyamond/SAM/res_200/qvi_res_200_t_29.nc')
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    for ii in range(len(lon)):
        if lon[ii]>180:
            lon[ii]-=360
    latlon = np.zeros((len(lat),len(lon)))
    for ii in range(len(lat)):
        for jj in range (192):
            latlon[ii,jj] = globe.is_ocean(lat[ii],lon[jj])
    return latlon

