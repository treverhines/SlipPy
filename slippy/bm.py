import numpy as np
from mpl_toolkits.basemap import Basemap

def geodetic_to_cartesian(pos_geo,basemap):
  ''' 
  Parameters
  ----------
    pos_geo : (...,D) array
      array of geodetic positions. The first and second component of 
      the last axis are longitude and latitude.  The last axis can 
      have additional components (e.g. height) and they will be 
      returned unaltered.

    basemap : Basemap instance    

  Returns
  -------
    pos_cart : (...,D) array
  '''    
  pos_cart = np.array(pos_geo,copy=True)
  pos_geo = np.asarray(pos_geo)
  pos_x,pos_y = basemap(pos_geo[...,0],pos_geo[...,1])
  pos_x = np.asarray(pos_x)
  pos_y = np.asarray(pos_y)
  pos_cart[...,0] = pos_x
  pos_cart[...,1] = pos_y
  return pos_cart

def cartesian_to_geodetic(pos_cart,basemap):
  ''' 
  Parameters
  ----------
    pos_cart : (...,D) array
      array of cartesian positions 

    basemap : Basemap instance
    
  Returns
  -------
    pos_geo : (...,D) array  
  '''
  pos_geo = np.array(pos_cart,copy=True)
  pos_cart = np.asarray(pos_cart)
  pos_lon,pos_lat = basemap(pos_cart[...,0],pos_cart[...,1],inverse=True)
  pos_lon = np.asarray(pos_lon)
  pos_lat = np.asarray(pos_lat)
  pos_geo[...,0] = pos_lon
  pos_geo[...,1] = pos_lat
  return pos_geo
  
def create_default_basemap(lon_lst,lat_lst,**kwargs):
  ''' 
  creates a basemap that bounds lat_lst and lon_lst
  '''
  if (len(lon_lst) == 0) | (len(lat_lst) == 0):
    return Basemap(projection='tmerc',
                   lon_0 = -90.0,
                   lat_0 = 41.0,
                   llcrnrlat = 26.0,
                   llcrnrlon = -128.0,
                   urcrnrlat = 48.0,
                   urcrnrlon = -53.0,
                   **kwargs) 
    
  lon_buff = (max(lon_lst) - min(lon_lst))/20.0
  lat_buff = (max(lat_lst) - min(lat_lst))/20.0
  if lon_buff < 0.2:
    lon_buff = 0.2

  if lat_buff < 0.2:
    lat_buff = 0.2

  llcrnrlon = min(lon_lst) - lon_buff
  llcrnrlat = min(lat_lst) - lat_buff
  urcrnrlon = max(lon_lst) + lon_buff
  urcrnrlat = max(lat_lst) + lat_buff
  lon_0 = (llcrnrlon + urcrnrlon)/2.0
  lat_0 = (llcrnrlat + urcrnrlat)/2.0
  return Basemap(projection='tmerc',
                 lon_0 = lon_0,
                 lat_0 = lat_0,
                 llcrnrlon = llcrnrlon,
                 llcrnrlat = llcrnrlat,
                 urcrnrlon = urcrnrlon,
                 urcrnrlat = urcrnrlat,
                 **kwargs)
  

