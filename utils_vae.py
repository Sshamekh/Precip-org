import numpy as np
from netCDF4 import Dataset



def normalize(data,method ):     
    if method =='std':
        mins = np.min(data)
        stds = np.std(data)
        data = (data - mins)/stds
    if method == 'minmax':
        mins = np.min(data)
        maxs = np.max(data)
        data = (data - mins)/maxs
    if method == 'None':
        print ('no scaling')
    
    return data

def normalize_mean(data,method = 'std'): 
    
    if method =='std':
        mins = (np.min(data,axis = (0,1,2))).reshape(1,1,1,-1)
        stds = (np.std(data,axis = (0,1,2))).reshape(1,1,1,-1)
        #data = (data - mins)/stds
        print (stds)
    if method == 'minmax':
        mins = (np.min(data,axis = (0,1,2))).reshape(1,1,1,-1)
        maxs = (np.max(data,axis = (0,1,2))).reshape(1,1,1,-1)
        data = (data - mins)/maxs
    if method == 'None':
        print ('no scaling')
    
    return data

def read_file(path,var,hr,ls):
    """path: link to the nc file. This file contains one variable in 3 courdinate: t,lat,lon
        var: the variable to process
        hr (high resolution): is the model original resolution 
        ls (largescale): is the new resolution we want to coarsegrain to 
    """
    lon1=0
    lon2=-1
    ds = Dataset(path)
    field = ds.variables[var][:,:,lon1:lon2]
    
    nt,nlat,nlon = field.shape
    times = ds.variables['time'][:]
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][lon1:lon2]
    nxy = int(ls/hr) # number of grids in each direction of the new grid
    ngrid_lat = int(nlat/nxy) # number of grids in latitudinal direction in coarse data
    ngrid_lon = int(nlon/nxy) # number of grids in llongitudindal direction in coarse data
    
    field = field[:, :ngrid_lat*nxy, :ngrid_lon*nxy] # cropping the field
    
    nt,nlat,nlon = field.shape
    newfield = field.reshape(nt , ngrid_lat,nxy,ngrid_lon,nxy)
    return newfield

def read_landmask(path,hr,ls):
    """path: link to land mask file. This file contains one variable in 3 courdinate: t,lat,lon
        hr (high resolution): is the model original resolution 
        ls (largescale): is the new resolution we want to coarsegrain to 
    """
    lon1=0
    lon2=-1
    
    field= np.load(path + 'landmasknegative.npy')[:,:,lon1:lon2]
    
    nt,nlat,nlon = field.shape
    nxy = int(ls/hr) # number of grids in each direction of the new grid
    ngrid_lat = int(nlat/nxy) # number of grids in latitudinal direction in coarse data
    ngrid_lon = int(nlon/nxy) # number of grids in llongitudindal direction in coarse data
    
    field = field[:, :ngrid_lat*nxy, :ngrid_lon*nxy] # cropping the field
    
    nt,nlat,nlon = field.shape
    newfield = field.reshape(nt , ngrid_lat,nxy,ngrid_lon,nxy)
    return newfield

def read_path(path,var,hr,ls,t1,t2):
    """ 
    make a loop over time steps(here we have only three). Read the field for all time steps, stac them and return
    """
    for i in range(t1,t2):
        
        path_i = path+var+'-'+str(i+1)+'.nc'
        field_i = read_file(path_i,var,hr,ls)
        if var == "pracc":
            field_i = field_i[1:,:]-field_i[:-1,:]
        else:
            field_i = field_i[1:,:]
        
        if i==t1 : 
            all_field = np.copy(field_i)
        else:
            all_field = np.concatenate((all_field,field_i),axis = 0)
            
    if var == "pracc":
        landmask = read_landmask(path,hr,ls) 
        all_field = all_field * landmask
   
    return np.moveaxis(all_field,2,3)
        
    

    
def read_field(path,varlist,hr,ls,t1,t2,method, data_reshape = False): 
    
    
    for i,var in enumerate (varlist):
        print (var)
        data = read_path(path,var,hr,ls,t1,t2)
        nt,nlat,nlon,nx,ny = data.shape
        if i==0:
            data = normalize(data,method)
            data_all = np.reshape(data,(nt,nlat,nlon,nx,ny,1))
        else: 
            data = normalize(data,method)
            data_all = np.concatenate (( data_all, np.reshape(data,(nt,nlat,nlon,nx,ny,1))), axis = -1)
    print (data_all.shape)
            
    
    if data_reshape == True : 
        nt,nlat,nlon,nx,ny,nfield = data_all.shape
        data_all = data_all.reshape(nt*nlat*nlon,nx,ny,nfield)
    return data_all


def read_field_mean(path,varlist,hr,ls,t1,t2,method = 'std', data_reshape = False): 
    
    
    for i,var in enumerate (varlist):
        print (var)
        data = read_path(path,var,hr,ls,t1,t2)
        nt,nlat,nlon,nx,ny = data.shape
        if i==0:
            data = np.mean(data,(-1,-2))
            data_all = np.reshape(data,(nt,nlat,nlon,1))
        else: 
            data = np.mean(data,(-1,-2))
            
            data_all = np.concatenate (( data_all, np.reshape(data,(nt,nlat,nlon,1))), axis = -1)
    print (data_all.shape)
            
    data_all = normalize_mean(data_all,method)
    
    if data_reshape == True : 
        nt,nlat,nlon,nx,ny,nfield = data_all.shape
        data_all = data_all.reshape(nt*nlat*nlon,nfield)
    return data_all


def train_test_data(path, hr,ls,t1,t2,traintest = 'train',threshold_precip = 0.1,mask_threshold = True): 

    varlist_inputs   = ['qvi','ts', 'tas','huss']
    varlist_inputs_2 = ['hfss','hfls']
    nvin_lg = len(varlist_inputs)
    nvin_lg_2 = len(varlist_inputs_2)
    varlist_hr = ['qvi']

    varlist_outputs = ['pracc']
    print ('Reading large-scale outputs')
    outputs = read_field_mean(path,varlist_outputs,hr,ls,t1,t2,method = 'std',data_reshape = False)
    nvout = len(varlist_outputs)
    print ('Reading high resolution input')
    inputs_hr = read_field(path,varlist_hr,hr,ls,t1,t2,method = 'None',data_reshape = False) 
    print ('Reading large-scale inputs')
    inputs = read_field_mean(path,varlist_inputs,hr,ls,t1,t2,method = 'minmax',data_reshape = False) 
    
    print ('Reading large-scale inputs2')
    inputs2 = read_field_mean(path,varlist_inputs_2,hr,ls,t1,t2,method = 'minmax',data_reshape = False) 
    

    #scaling values
    sstmax = 330
    pwmax = 73
    #precipmax = #7
    
    #scaling high res fields
    inputs_hr[:,:,:,:,:,0] /= pwmax

    
    #reshaping data 
    nt,nlat,nlon,nxy,nxy,ff = inputs_hr.shape
    print ('nt, nlat, nlon, nxy, nxy, nv_hr:',inputs_hr.shape)
    inputs_hr = inputs_hr.reshape(nt*nlat*nlon,nxy,nxy,ff)
    outputs = outputs.reshape(nt*nlat*nlon,nvout)

    x_lg = inputs.reshape(nt*nlat*nlon,nvin_lg)
    x_lg2 = inputs2.reshape(nt*nlat*nlon,nvin_lg_2)
    
    # remove grids with small or no precip
    if mask_threshold:
        inds_train = np.where(outputs[:,-1]>threshold_precip)[0]
        x_hr = inputs_hr[inds_train,:]
        x_lg = x_lg[inds_train,:]
        x_lg2 = x_lg2[inds_train,:]
        y_lg = outputs[inds_train,:]
        #scale precip
        #y_train_lg[:,-1]/=precipmax 
    else: 
        x_hr = inputs_hr
        x_lg = x_lg
        y_lg = outputs
        #scale precip
    
    pw_anomaly= (x_hr - np.mean(x_hr,axis = (1,2),keepdims = True)) 
    x_lg = np.concatenate((x_lg,x_lg2),axis = -1)
    
    total_sample = x_lg.shape[0]
    nt_train = int(0.8*total_sample)
    print ('nt_train is: ', nt_train)
    
    x_train_hr,x_test_hr = pw_anomaly[:nt_train,:],pw_anomaly[nt_train:,:]

    x_train_lg,x_test_lg = x_lg[:nt_train,:],x_lg[nt_train:,:]

    y_train_lg,y_test_lg = y_lg[:nt_train,:],y_lg[nt_train:,:]
        
    
    return x_train_hr , x_train_lg , y_train_lg ,x_test_hr , x_test_lg , y_test_lg

def read_data_for_z(path, hr,ls,t1,t2,traintest = 'train',threshold_precip = 0.1,mask_threshold = True):  
    varlist_inputs = ['qvi','ts', 'tas','huss']
    varlist_inputs_2 = ['vas','uas', 'hfss','hfls']
    nvin_lg = len(varlist_inputs)
    
    varlist_hr = ['qvi']

    varlist_outputs = ['pracc']
    nvout = len(varlist_outputs)
    print ('Reading high resolution input')
    inputs_hr = read_field(path,varlist_hr,hr,ls,t1,t2,method = 'None',data_reshape = False) 
    print ('Reading large-scale inputs')
    inputs = read_field_mean(path,varlist_inputs,hr,ls,t1,t2,method = 'minmax',data_reshape = False) 
    print ('Reading large-scale inputs2')
    inputs2 = read_field_mean(path,varlist_inputs_2,hr,ls,t1,t2,method = 'minmax',data_reshape = False) 
    print ('Reading large-scale outputs')
    outputs = read_field_mean(path,varlist_outputs,hr,ls,t1,t2,method = 'std',data_reshape = False)

    #scaling values
    sstmax = 330
    pwmax = 73
    #precipmax = 7
    
    #scaling high res fields
    inputs_hr[:,:,:,:,:,0] /= pwmax
    
    return inputs_hr,inputs,inputs2,outputs

