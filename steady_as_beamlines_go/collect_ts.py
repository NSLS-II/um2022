import numpy as np
from photutils.centroids import centroid_com
import pandas as pd
from anomaly.extract_features import get_features_single_datum

# from tiled.client import from_profile
# csx = from_profile("csx")

from tiled.client import from_uri
c = from_uri("https://tiled-demo.blueskyproject.io/api")
csx = c["csx"]["raw"]

def process_data_serial(fov):
    
    # calculate time series of std withing current_ROI 
    std_ts = np.nanstd(fov, axis = (1,2), )
    
    # calculate intensity time series for current_ROI
    intensity_ts = np.nanmean(fov, axis = (1,2))
    
        
    # calculate of center of mass time series
    com_address = np.array([np.flip(centroid_com(im)) for im in fov]) 
    com_x_ts = com_address[:, 0]
    com_y_ts = com_address[:, 1]   
    
    #calculate spot shape parameters
    projected_x = np.nansum(fov, axis = 1)
    values = np.arange(projected_x.shape[1])
    norm = np.sum(projected_x, axis = 1).reshape(-1,1)
    com_x = np.sum((projected_x * values / norm ), axis = 1).reshape(-1,1)
    sigma_x = ((np.sum((values-com_x)**2, axis = 1)/projected_x.shape[1])**0.5)
    #adjusted_com_x = (values-com_x)/(sigma_x).reshape(-1,1)
    #skew_x_ts = np.sum(((adjusted_com_x**3)*projected_x)/ norm, axis = 1)
    #kurtosis_x_ts = np.sum(((adjusted_com_x**4)*projected_x)/ norm, axis = 1)
    sigma_x_ts = sigma_x
    
    projected_y = np.nansum(fov, axis = 2)
    values = np.arange(projected_y.shape[1])
    norm = np.sum(projected_y, axis = 1).reshape(-1,1)
    com_y = np.sum((projected_y * values / norm), axis = 1).reshape(-1,1)
    sigma_y = ((np.sum((values-com_y)**2, axis = 1) / projected_y.shape[1])**0.5)
    #adjusted_com_y = (values-com_y)/(sigma_y).reshape(-1,1)
    #skew_y_ts = np.sum(((adjusted_com_y**3)*projected_y) / norm, axis = 1)
    #kurtosis_y_ts = np.sum(((adjusted_com_y**4)*projected_y) / norm, axis = 1)
    sigma_y_ts = sigma_y
    
    return (std_ts, intensity_ts, com_x_ts, com_y_ts, sigma_x_ts, sigma_y_ts)
    
def package_data(ts_data,  roi_name, classification_label, ):#scan_no):
    
    std_ts, intensity_ts, com_x_ts, com_y_ts, sigma_x_ts, sigma_y_ts =  list(ts_data)
    
    if classification_label is None:
        classification_label = 'unknown'
    
    data = {'std_ts': std_ts,
            'intensity_ts': intensity_ts,
            'com_x_ts': com_x_ts,
            'com_y_ts': com_y_ts,
            'sigma_x_ts': sigma_x_ts,
            'sigma_y_ts': sigma_y_ts,
            'roi_name': roi_name,
            'classification_label': classification_label,
            #'scan_no': scan_no,        
           }
    
    return data
    
def get_data(time_series_arr, roi_name, classification_label = None):#,scan_no=None):
    
    if len(time_series_arr.shape) != 3:
        print('Make shape = (frames, pix, pix)')
        raise
    
    time_series_results =  process_data_serial(time_series_arr)
    
    ts_data = package_data(time_series_results, roi_name, classification_label,)# scan_no) 
    
    return ts_data
    
def make_input_array(images, Vstart, Hstart, Vsize, Hsize):
    
    Vend, Hend = int(Vstart+Vsize), int(Hstart+Hsize)
    array_ts = np.copy(images[:, Vstart:Vend, Hstart:Hend])
    
    return array_ts
    

def prep_model_input(images, roi_name_coord, scan_no, classification_label):
    roi_name, roi_coords  = roi_name_coord
    Vpix, Hpix,  Vsize, Hsize = roi_coords
    
    #print(roi_name)
    #print(roi_coords)
    input_arr = make_input_array(images, Vpix, Hpix, Vsize, Hsize)
    data_dict = get_data(input_arr, f'{scan_no}_{roi_name}',  classification_label)
    series = pd.Series(data_dict)
    features = get_features_single_datum(series)
    df = pd.DataFrame([features])
    new_input_data = (df.drop(columns=["target", "roi"]))
    
    return new_input_data, data_dict, f'{scan_no}_{roi_name}'
            
def get_images_from_tiled(run_data):
    #run = csx[scan]['primary']['data']['dif_beam_hdf5_image'][:, :, 400:1200, :1200].compute()
    images = run_data.to_numpy()
    _, fs, vpix, hpix = images.shape
    images = images.reshape(fs, vpix, hpix)
    
    return images


