> Created on Thu Jul  9 22/10/09 2020  @author: Richie Bao-caDesign (cadesign.cn)

## 1 Kernel density estimation and geospatial point density distribution
### 1.1  Kernel density estimation
#### 1.1.1 Kernel density estimation for a single variable (a one-dimensional array)
When the histogram's spacing is reduced to the limit, a curve can be fitted, and the formula to calculate the distribution curve is the Probability Density Function(PDF); this is explained in the [normal distribution section](https://richiebao.github.io/Urban-Spatial-Data-Analysis_python/#/./notebook_code/normalDis_PDF_outliers), and the nonparametric method used to estimate the probability density function is the Kernel Density Estimation(KDE). Kernel Density Estimation is a fundamental data smoothing problem. For example, for an uneven histogram, a kernel K is given, and bandwidth is specified, the Kernel Density Estimation is defined as:  $\hat{f} _{n} (x)= \frac{1}{nh}  \sum_{i=1}^n {K( \frac{x- x_{i} }{h} )} $，Where K is the kernel function, $h$ is the bandwidth. And there are multiple kernel functions. For example, the Gaussian Kernel is:$\frac{1}{ \sqrt{2 \pi } }  e^{- \frac{1}{2}  x^{2} } $，Put it into the Kernel Density Estimation formula, and the result is:$\hat{f} _{n} (x)= \frac{1}{ \sqrt{2 \pi } nh}  \sum_{i=1}^n { e^{ -\frac{ (x- x_{i} )^{2} }{2 h^{2} } } } $。

Three curves are drawn in the following code. The bold red line is the probability density function. Both thin lines are kernel density estimation(Gaussian kernel), but the blue line is coded directly according to the kernel density formula, and set bandwidth h=0.4. The green line calculates the Gaussian kernel density estimation using the `stats.gaussian_kde()` method under the Scipy library.

> Nonparametric Statistics is a branch of Statistics, but it is not entirely based on the parameterized probability distribution, such as through the parameters of the mean and variance (or standard deviation) to define a normal distribution, Nonparametric Statistics based on distribution-free or specified distribution but not given the distribution parameters, such as when dealing with PDF generally cannot be given parameters are classified as the normal distribution. The basic idea is to use data to make inferences about an unknown quantity with as few assumptions as possible, which usually means using statistical models with infinite dimensions.

> For kernel density estimation, the term density can be visualized as the small olive green vertical lines' distribution density in the figure below. 

Reference: Wikipedia, and Larry Wasserman.All of nonparametric statistics.Springer (October 21, 2005); Urmila Diwekar,Amy David.Bonus algorithm for large scale stochastic nonlinear programming problems.Springer; 2015 edition (March 5, 2015)

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
x=np.linspace(stats.norm.ppf(0.001,loc=0,scale=1),stats.norm.ppf(0.999,loc=0,scale=1), 100) #A value with an equal probability of between 0.1% and 99.9%. If the parameters loc and scale are not given, the default is the standard normal distribution, namely, loc=0, scale=1.
pdf=stats.norm.pdf(x)

plt.figure(figsize=(25,5))
plt.plot(x,pdf,'r-', lw=5, alpha=0.6, label='norm_pdf')

random_variates=stats.norm.rvs(loc=0,scale=1,size=500)
count, bins, ignored =plt.hist(random_variates,bins=100,density=True,histtype='stepfilled',alpha=0.2)
plt.eventplot(random_variates,color='y',linelengths=0.03,lineoffsets=0.025) #Draw short lines for a given position

rVar_sort=np.sort(random_variates)
h=0.4 #bandwidth,bw
n=len(rVar_sort)
kde_Gaussian=[sum(math.exp(-1*math.pow(vi-vj,2)/(2*math.pow(h,2))) for vj in rVar_sort)/(h*n*math.sqrt(2*math.pi)) for vi in rVar_sort] #The above Gaussian kernel density estimation formula is converted into code.
plt.plot(rVar_sort,kde_Gaussian,'b-', lw=2, alpha=0.6, label='kde_formula,h=%s'%h)

scipyStatsGaussian_kde=stats.gaussian_kde(random_variates)
plt.plot(bins,scipyStatsGaussian_kde(bins),'g-', lw=2, alpha=0.6, label='scipyStatsGaussian_kde')
plt.legend()
plt.show()
```

<a href=""><img src="./imgs/5_1.png" height="auto" width="auto" title="caDesign"></a>

Bandwidth affects the degree of smoothness. The following experiment sets different values to observe the changes in the kernel density curve. Concerning the optimal bandwidth inference, <em>Bonus algorithm for large scale stochastic nonlinear programming problems</em>Chapter 3，<em>Probability Density Function and Kernel Density Estimation</em>, One method is mentioned in the section, and you can refer to it.

```python
bws=np.arange(0.1,1,0.2)
colors_kde=['C{}'.format(i) for i in range(len(bws))] #maplotlib，Specify the color
i=0
plt.figure(figsize=(25,5))
for h in bws:
    kde_Gaussian=[sum(math.exp(-1*math.pow(vi-vj,2)/(2*math.pow(h,2))) for vj in rVar_sort)/(h*n*math.sqrt(2*math.pi)) for vi in rVar_sort] #The above Gaussian kernel density estimation formula is converted into code.
    plt.plot(rVar_sort,kde_Gaussian,color=colors_kde[i], lw=2, alpha=0.6, label='kde_formula,h=%.2f'%h)
    i+=1
plt.legend()
plt.show()
```

<a href=""><img src="./imgs/5_2.png" height="auto" width="auto" title="caDesign"></a>

#### 1.1.2 Kernel density estimation for multivariate(multidimensional arrays)
The kernel density estimation can smooth multidimensional data. For example, the production of the thermal diagram is two-dimensional smoothness based on the kernel density estimation. Taking Baidu POI and OSM data crawled as an example, `scipy.stats.gaussian_kde()`is directly used to calculate the kernel density. The second is to extract the line classified by 'delicacy' to calculate delicacy's kernel density estimation.

```python
import pandas as pd
poi_gpd=pd.read_pickle('./data/poiAll_gpd.pkl') #Read the POI data already stored in .pkl format, including the 'geometry' field, as the GeoDataFrame geographic information data, which can be quickly viewed through 'poi_gpd.plot()'. 
poi_gpd.plot(marker=".",markersize=5) #Check to see if the POI data is read properly.
```

<a href=""><img src="./imgs/5_3.png" height="auto" width="auto" title="caDesign"></a>

* Kernel density estimation of all POI point data and establishment of maps

```python
from scipy import stats
poi_coordinates=poi_gpd[['location_lng','location_lat']].to_numpy().T  #The array structure is determined according to the 'stats.gaussian_kde()' input parameter
poi_coordi_kernel=stats.gaussian_kde(poi_coordinates) #Kernel desity estimation
poi_gpd['poi_kde']=poi_coordi_kernel(poi_coordinates)

import plotly.express as px
poi_gpd.detail_info_price=poi_gpd.detail_info_price.fillna(0) 
mapbox_token='pk.eyJ1IjoicmljaGllYmFvIiwiYSI6ImNrYjB3N2NyMzBlMG8yc254dTRzNnMyeHMifQ.QT7MdjQKs9Y6OtaJaJAn0A'
px.set_mapbox_access_token(mapbox_token)
fig=px.scatter_mapbox(poi_gpd,lat=poi_gpd.location_lat, lon=poi_gpd.location_lng,color='poi_kde',color_continuous_scale=px.colors.sequential.PuBuGn, size_max=15, zoom=10) #You can also select columns to increase the display information by configuring 'size=""'.
fig.show()

poi_gpd.head()
```

<a href=""><img src="./imgs/5_4.png" height="auto" width="auto" title="caDesign"></a>

|index|name|	location_lat|	location_lng|	detail_info_tag|	detail_info_overall_rating	|detail_info_price|	geometry|	poi_kde|
|---|---|---|---|---|---|---|---|---|
|0|	御荷苑饭店|	34.182148|	108.823310|	美食;中餐厅|	4.0	|0|	POINT (108.82331 34.18215)	|2.917734|
|2|	一品轩餐厅|	34.183155|	108.823328|	美食;中餐厅	|5.0|	0	|POINT (108.82333 34.18316)	|3.131301|
|4|	老米家泡馍|	34.183547|	108.823851|	美食;中餐厅	|5.0|	0	|POINT (108.82385 34.18355)	|3.264152|
|6|	关中印象咥长安(创汇店)|	34.183542|	108.823498|	美食;中餐厅|	4.5|	8	|POINT (108.82350 34.18354)|	3.227471|
|8|	惠记葫芦头梆梆肉|	34.183534|	108.823589|	美食;中餐厅|	4.6|	0|	POINT (108.82359 34.18353)	|3.234969|

* Kernel density estimation for 'delicacy' location point distribution

```python
pd.options.mode.chained_assignment = None
poi_gpd['tag_primary']=poi_gpd['detail_info_tag'].apply(lambda row:str(row).split(";")[0])
poi_classificationName={
        "美食":"delicacy",
        "酒店":"hotel",
        "购物":"shopping",
        "生活服务":"lifeService",
        "丽人":"beauty",
        "旅游景点":"spot",
        "休闲娱乐":"entertainment",
        "运动健身":"sports",
        "教育培训":"education",
        "文化传媒":"media",
        "医疗":"medicalTreatment",
        "汽车服务":"carService",
        "交通设施":"trafficFacilities",
        "金融":"finance",
        "房地产":"realEstate",
        "公司企业":"corporation",
        "政府机构":"government",
        'nan':'nan'
        }
poi_gpd.loc[:,["tag_primary"]]=poi_gpd["tag_primary"].replace(poi_classificationName)
delicacy_df=poi_gpd.loc[poi_gpd.tag_primary=='delicacy']
delicacy_coordi=delicacy_df[['location_lng','location_lat']].to_numpy().T 
delicacy_kernel=stats.gaussian_kde(delicacy_coordi) #Kernel density estimation
delicacy_df['delicacy_kde']=delicacy_kernel(delicacy_coordi)

import plotly.express as px
poi_gpd.detail_info_price=poi_gpd.detail_info_price.fillna(0) 
mapbox_token='pk.eyJ1IjoicmljaGllYmFvIiwiYSI6ImNrYjB3N2NyMzBlMG8yc254dTRzNnMyeHMifQ.QT7MdjQKs9Y6OtaJaJAn0A'
px.set_mapbox_access_token(mapbox_token)
fig=px.scatter_mapbox(delicacy_df,lat=delicacy_df.location_lat, lon=delicacy_df.location_lng,color='delicacy_kde',color_continuous_scale=px.colors.sequential.PuBuGn, size_max=15, zoom=10) #You can also select columns to increase the display information by configuring 'size=""'.
fig.show()
```

<a href=""><img src="./imgs/5_5.png" height="auto" width="auto" title="caDesign"></a>

### 1.2 Kernel density estimation results are converted to geographic raster data
The final calculated value of the above two-dimensional geospatial points' kernel density estimation falls on the point position itself on the map expression.  Suppose you want to display the estimated value in the form of the raster. In that case, you can convert it into raster data, which is essentially the geospatial point data to raster data. In the following methods, two transformations are provided. One way is to store GeoDataFrame format data with kernel density estimation using `gdf.to_file()` to save point data in .shp format and then define a function that converts .shp to a raster. The second approach directly defines a function that can now compute the kernel density estimation and save it directly as raster data. In both cases, it is necessary to define a coordinate projection of data in GeoDataFrame format and extract its information for extracting or defining coordinate projections using the GDAL library, so there must be a common coordinate system, Geopandas library, and GDAL library, that can be cross-transformed, and the EPSG numbered system is a good choice. European Petroleum Survey Group(EPSG) was the first to establish this coding system. The most important codes include:

EPSG:4326 - WGS84, widely used in the map and navigation systems, in geographical spatial data analysis, the geographic coordinate system is the most basic characterization of the location data coordinate system. Generally, all kinds of geospatial data information take WGS84 as the basic coordinate system. The projection can then be configured on different data types or platforms and according to the different analysis purposes, especially the changes of actual geographical locations.

EPSG:3857 - Pseudo Mercator projection, also known as sphere Mercator, is used to display many web-based mapping tools, including Google Maps and OpenStreetMap.

For EPSG coding, see:[spatialreference](https://spatialreference.org/)，和[epsg.io](https://epsg.io/)

The output raster's coordinate projection system in this experiment is EPSG:32649, namely, WGS 84 / UTM zone 49N, which corresponds to the coordinate projection system used by Landsat remote sensing images in the Xi'an region. Usually, when analyzing a certain urban space problem, many data types from different sources may be used. The most basic WGS84 shall be taken as the data storage coordinate system as far as possible. The display of data often needs the corresponding region's coordinate system to optimize the display result and make it more suitable for reading.

```python
import geopandas as gpd
from pyproj import CRS
import os
import pandas as pd

poi_gpd=pd.read_pickle('./data/poiAll_gpd.pkl') #ead the POI data already stored in .pkl format, including the 'geometry' field, as the GeoDataFrame geographic information data, which can be quickly viewed through 'poi_gpd.plot()'. 
poi_gpd.plot(marker=".",markersize=5) #Check to see if the POI data is read properly.

print("original projection:",poi_gpd.crs)
poi_gpd_copy=poi_gpd.copy(deep=True)
poi_gpd_copy=poi_gpd_copy.to_crs(CRS("EPSG:32649"))
print("re-projecting:",poi_gpd_copy.crs)

save_path=r'./data/geoData'
poi_epsg32649_fn='poi_epsg32649.shp'
poi_gpd_copy.to_file(os.path.join(save_path,poi_epsg32649_fn))
```

```
original projection: epsg:4326
re-projecting: EPSG:32649
```

<a href=""><img src="./imgs/5_6.png" height="auto" width="auto" title="caDesign"></a>

#### 1.2.1 .shp format geospatial point data to raster data
At the heart of this function is the `gdal.RasterizeLayer()` method provided by the GDAL library, which writes the values of .shp point layer fields to the corresponding raster location, thus avoiding the need to write code for the array of raster cell location and field values. On the definition of raster projection, the coordinate projection system is extracted from .shp point data. The code position is after the data has been written to the raster cell, and if it is before, a coordinate projection error occurs. The null value is usually set to -9999. GDAL provides the raster definition by getting the raster driver `gdal.GetDriverByName('GTiff')`, then establishing `.Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64)`, and configure the geographic transform `target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))`, read the raster band `band=target_ds.GetRasterBand(1)`, and write the value `band.WriteArray()` to it to complete the definition of the raster. In geographic transformation, because the map is usually northward up, parameters 3 and 5 are usually configured as 0.

```python
#convert points .shp to raster. use raster.SetGeoTransform, Rasterize the data. Refer to the official GDAL code
def pts2raster(pts_shp,raster_path,cellSize,field_name=False):
    from osgeo import gdal, ogr,osr
    '''
    function - Convert point data in .shp format to .tif raster
    
    Paras:
    pts_shp - point data file path in .shp format
    raster_path - Save the raster file path
    cellSize - Raster cell size
    field_name - .shp point data attribute field writing to the raster
    '''
    #Define NoData value of new raster
    NoData_value=-9999
    
    #Open the .shp data source and read in the extent
    source_ds=ogr.Open(pts_shp)
    source_layer=source_ds.GetLayer()
    x_min, x_max, y_min, y_max=source_layer.GetExtent()
    
    #Create the destination data source
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)
    target_ds=gdal.GetDriverByName('GTiff').Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64) #create(filename,x_size,y_size,band_count,data_type,creation_options)。gdal data type, gdal.GDT_Float64,gdal.GDT_Int32...
    target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)

    #Writes data to the raster layer
    if field_name:
        gdal.RasterizeLayer(target_ds,[1], source_layer,options=["ATTRIBUTE={0}".format(field_name)])
    else:
        gdal.RasterizeLayer(target_ds,[1], source_layer,burn_values=[-1])   
        
    #Configure the projected coordinate system
    spatialRef=source_layer.GetSpatialRef()
    target_ds.SetProjection(spatialRef.ExportToWkt())       
        
    outband.FlushCache()
    return gdal.Open(raster_path).ReadAsArray()

pts_shp=os.path.join(save_path,poi_epsg32649_fn)
raster_path=os.path.join(save_path,r'poi_epsg32649.tif')
cellSize=100
field_name='poi_kde'
poiRaster_array=pts2raster(pts_shp,raster_path,cellSize,field_name)
print("conversion complete!")
```

```
conversion complete!
```

Library in python that handles raster data can use [rasterio](https://rasterio.readthedocs.io/en/latest/quickstart.html#reading-raster-data),the method is also more convenient than GDAL, which can significantly reduce the amount of code. Use the library to read information about the raster data, read the data as an array, and print the raster to view it.

```python
raster_path=os.path.join(save_path,r'poi_epsg32649.tif')
import rasterio
dataset=rasterio.open(raster_path)
print(
    "band count:",dataset.count,'\n', #Check the number of raster bands
    "columns wide:",dataset.width,'\n', #Check the raster width.
    "rows hight:",dataset.height,'\n', #Check the raster height.
    "dataset's index and data type:",{i: dtype for i, dtype in zip(dataset.indexes, dataset.dtypes)},'\n',#View the band and its data type
    "bounds:",dataset.bounds,'\n', #View the coordinates of the lower left and upper right corner of the enclosing rectangle boundary.
    "geospatial transform:",dataset.transform,'\n', #Geospatial transformation of datasets
    "lower right corner:",dataset.transform*(dataset.width,dataset.height),'\n', #Calculate the coordinates of the lower right corner of the enclosing rectangle boundary.
    "crs:",dataset.crs,'\n', #The geographic coordinate projection system
    "band's index number:",dataset.indexes,'\n' #Raster layer(band) index
    )
band1=dataset.read(1) #Read the raster band data as an array.
print(band1)
```

```
band count: 1 
 columns wide: 653 
 rows hight: 446 
 dataset's index and data type: {1: 'float64'} 
 bounds: BoundingBox(left=294142.8129965692, bottom=3783958.0837052986, right=326792.8129965692, top=3806258.0837052986) 
 geospatial transform: | 50.00, 0.00, 294142.81|
| 0.00,-50.00, 3806258.08|
| 0.00, 0.00, 1.00| 
 lower right corner: (326792.8129965692, 3783958.0837052986) 
 crs: EPSG:32649 
 band's index number: (1,) 

[[-9999. -9999. -9999. ... -9999. -9999. -9999.]
 [-9999. -9999. -9999. ... -9999. -9999. -9999.]
 [-9999. -9999. -9999. ... -9999. -9999. -9999.]
 ...
 [-9999. -9999. -9999. ... -9999. -9999. -9999.]
 [-9999. -9999. -9999. ... -9999. -9999. -9999.]
 [-9999. -9999. -9999. ... -9999. -9999. -9999.]]
 ```

```python
from rasterio.plot import show
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
show((dataset,1),cmap='Greens') 
plt.show()
```

<a href=""><img src="./imgs/5_7.png" height="auto" width="auto" title="caDesign"></a>

Rasterio library supported colors(basically the same as the Matplotlib library; you can see the color band's specific name in this library): 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'

Open it in QGIS or ArcGIS to see the results. The rasterization process of this part begins with the null value configured as `outband.SetNoDataValue(-9999)`, later, when writing data to the raster, only the position containing the value replaces the original null value of the seen raster is partially transparent.

<a href=""><img src="./imgs/5_8.jpg" height="auto" width="auto" title="caDesign"></a>

#### 1.2.2 Given geospatial point data in GeoDataFrame format, compute kernel density estimation, and save as raster data
You can pass in GeoDataFrame format geospatial point data(pts_geoDF), given save location (raster_path), cell size(cellSize), and adjust the kernel density estimation by scaling factor(scale) to get the final kernel density estimation raster(thermal map), this method of calculation reduces the need for intermediate steps, especially without first converting point data to .shp format and then storing as raster data. But this place multiple steps in the calculation of a function will be 'one step' of the computing time stretched, so it is best in function through the 'print()' function to print complete information, avoid large quantities of data calculation, do not know the completed progress, unable to determine whether the program is still in normal operation, or has been completed, even was suspended.

The GDAL library provides a method `outband.WriteArray() ` for creating a raster and writing values to the raster cells. The passed parameter is an array, which corresponds to the raster position. Therefore, after calculating the kernel density estimation, it is necessary to redefine the position coordinate of extracting the estimated value, instead of using the point coordinates directly. In position definition, with the aid of `np.meshgrid()` implementation, also need to pay attention to the position estimation from the above definition; the order is rebellious, namely read line by line from down to up, this conforms to the definition of image pixel sequence, but in the geographic raster data is usually written down, so use `np.flip(Z,0)` flip array, the last line will be moved to the positive first line, the penultimate line moved to the positive second line, etc.

```python
def pts_geoDF2raster(pts_geoDF,raster_path,cellSize,scale):
    from osgeo import gdal,ogr,osr
    import numpy as np
    from scipy import stats
    '''
    function - Convert point data in GeoDaraFrame format to raster data
    
    Paras:
    pts_geoDF -  Point data in GeoDaraFrame format
    raster_path - Save the raster file path.
    cellSize - Raster cell size
    scale - Scale the kernel density estimation
    '''
    #Define NoData value of new raster
    NoData_value=-9999
    x_min, y_min,x_max, y_max=pts_geoDF.geometry.total_bounds

    #Create the destination data source using GDAL
    x_res=int((x_max - x_min) / cellSize)
    y_res=int((y_max - y_min) / cellSize)
    target_ds=gdal.GetDriverByName('GTiff').Create(raster_path, x_res, y_res, 1, gdal.GDT_Float64 )
    target_ds.SetGeoTransform((x_min, cellSize, 0, y_max, 0, -cellSize))
    outband=target_ds.GetRasterBand(1)
    outband.SetNoDataValue(NoData_value)   
    
    #Configure the projected coordinate system
    spatialRef = osr.SpatialReference()
    epsg=int(pts_geoDF.crs.srs.split(":")[-1])
    spatialRef.ImportFromEPSG(epsg)  
    target_ds.SetProjection(spatialRef.ExportToWkt())
    
    #Write data to the raster layer(band)
    #print(x_res,y_res)
    X, Y = np.meshgrid(np.linspace(x_min,x_max,x_res), np.linspace(y_min,y_max,y_res)) #Defines an array of raster cell coordinates that extract kernel density estimation
    positions=np.vstack([X.ravel(), Y.ravel()])
    values=np.vstack([pts_geoDF.geometry.x, pts_geoDF.geometry.y])    
    print("Start calculating kde...")
    kernel=stats.gaussian_kde(values)
    Z=np.reshape(kernel(positions).T, X.shape)
    print("Finish calculating kde!")
    #print(values)
        
    outband.WriteArray(np.flip(Z,0)*scale) #You need to flip the array and write the raster cells        
    outband.FlushCache()
    print("conversion complete!")
    return gdal.Open(raster_path).ReadAsArray()
    
save_path=r'./data/geoData'
raster_path_gpd=os.path.join(save_path,r'poi_gpd.tif')
cellSize=500 #The smaller the 'cellSize' value, the longer it will take to compute, and when you start debugging, you can try to scale it up to save computation time, such as a value of 50,100, etc.
scale=10**10 #Equivalent to math.pow(10,10)
poiRasterGeoDF_array=pts_geoDF2raster(poi_gpd_copy,raster_path_gpd,cellSize,scale)    
```

```
Start calculating kde...
Finish calculating kde!
conversion complete!
```

```python
save_path=r'./data/geoData'
raster_path_gpd=os.path.join(save_path,r'poi_gpd.tif')
import rasterio
dataset_gpd=rasterio.open(raster_path_gpd)

from rasterio.plot import show
import matplotlib.pyplot as plt
plt.figure(figsize=(15,10))
show((dataset_gpd,1),contour=True,cmap='Greens') #Turn on contour mode.
plt.show()
```

<a href=""><img src="./imgs/5_9.png" height="auto" width="auto" title="caDesign"></a>

```python
plt.figure(figsize=(15,10))
show((dataset_gpd,1),cmap='Greens') #Turn on contour mode.
plt.show()
```

<a href=""><img src="./imgs/5_10.png" height="auto" width="auto" title="caDesign"></a>

#### 1.2.3 Kernel density estimation of OSM geospatial point data for the City of Chicago and its region

Put the above two functions of .shp point to raster and the kernel density estimation calculation and then save as a raster in 'util.py' tool file for easy invocation.  in [OSM data processing](https://richiebao.github.io/Urban-Spatial-Data-Analysis_python/#/./notebook_code/OSM_dataProcessing) chapter has described the OSM data processing method, and gives a brief overview of the OSM geographical spatial structure of a data set. OSM node tag classification is rich, for different problems can be extracted for different tags for analysis. Because 'amenity' is associated with People's daily habitats, including livelihood, education, transportation, finance, health, entertainment, arts, and culture, there are more than 100 different categories. Therefore, multiple spatial point data tagged as 'amenity' are extracted, kernel density estimation is calculated, and its distribution is checked.

```python
import util 
start_time=util.start_time()
read_node_gdf=gpd.read_file("./data/osm_node.gpkg")
util.duration(start_time)
```

```
start time: 2020-07-12 14:53:32.200039
end time: 2020-07-12 15:29:14.515908
Total time spend:35.70 minutes
```

```python
read_node_gdf.head()
```

|index|type|	id|	version|	visible|	ts|	uid|	user|	changeet|	tagLen|	tags|	geometry|
|---|---|---|---|---|---|---|---|---|---|---|---|
|0|	node|	219850|	55|	True|	2018-02-20T05/50/28	|0|	|	0|	2|	{"highway": "motorway_junction", "ref": "276C"}|	POINT (-87.91012 41.75859)|
|1|	node|	219851|	48|	True|	2018-02-20T05/50/29	|0|	|	0|	2|	{"highway": "motorway_junction", "ref": "277A"}	|POINT (-87.90764 41.75931)|
|2|	node|	219966|	5|	True|	2009-04-04T22/47/50	|0|	|	0|	0|	{}|	POINT (-87.91596 43.01149)|
|3|	node|	219968|	12|	True|	2015-08-04T05/38/49	|0|	|	0|	2|	{"ref": "73B", "highway": "motorway_junction"}|	POINT (-87.92464 43.05606)|
|4|	node|	219969|	6|	True|	2009-04-14T00/13/37	|0|	|	0|	0|	{}|	POINT (-87.92441 43.05684)|

```python
amenity_poi=read_node_gdf[read_node_gdf.tags.apply(lambda row: "amenity" in eval(row).keys())] #Extract the tags column, including all lines fo the tag 'amenity.'
print("Finished amenity extraction!")
```

```
Finished amenity extraction!
```

Since it takes about 35 minutes for the original OSM data to be read, the extracted lines containing only 'amenity' reduced the data size significantly, and subsequent analysis only focused on the extracted data, so the data was saved separately to facilitate subsequent reading and analysis and avoid the time cost caused by reading large files. 

```python
print(
    "the overal data number:",read_node_gdf.shape,'\n',
    "the amenity data number:",amenity_poi.shape,'\n',     
     )
amenity_poi.to_file("./data/geoData/amenity_poi.gpkg",driver='GPKG')
print("Finished saving!")
```

```
the overal data number: (23859111, 11) 
 the amenity data number: (38421, 11) 

Finished saving!
```

```python
import geopandas as gpd
amenity_poi=gpd.read_file("./data/geoData/amenity_poi.gpkg")
amenity_poi.plot(marker=".",markersize=5,figsize=(15, 8))
```

<a href=""><img src="./imgs/5_11.png" height="auto" width="auto" title="caDesign"></a>

* The first method is used to calculate the kernel density estimate and save it as raster data.

Note that it is possible to cut the field name when defining the field name if you write data in .shp format. For example, if the definition field name is “amenity_kde”, then the field name might be truncated to "amenity_kd" after saving it as .shp with geopandas library. If you read the data without noticing changes in the field name, you may cause a hidden error.

```python
import pandas as pd
import numpy as np
from scipy import stats
pd.options.mode.chained_assignment = None

start_time=util.start_time()
poi_coordinates=np.array([amenity_poi.geometry.x,amenity_poi.geometry.y])
amenity_kernel=stats.gaussian_kde(poi_coordinates) #Kernel density estimation
amenity_poi['amenityKDE']=amenity_kernel(poi_coordinates) 
util.duration(start_time)
```

```
start time: 2020-07-12 15:45:02.394814
end time: 2020-07-12 15:45:25.106516
Total time spend:0.37 minutes
```

```python
import geopandas as gpd
from pyproj import CRS
import os
print("original projection:",amenity_poi.crs)
amenity_poi_copy=amenity_poi.copy(deep=True)
amenity_poi_copy=amenity_poi.to_crs(CRS("EPSG:32616"))  #EPSG:32616 - WGS 84 / UTM zone 16N - Projected
print("re-projecting:",amenity_poi_copy.crs)

amenity_kde_fn='./data/geoData/amenity_kde.shp'
amenity_poi_copy.to_file(amenity_kde_fn)

raster_path=r'./data/geoData/amenity_kde.tif'
cellSize=300
field_name='amenityKDE' 
amenityKDE_array=util.pts2raster(amenity_kde_fn,raster_path,cellSize,field_name)
print("finished kde computing.")
```

```
original projection: epsg:4326
re-projecting: EPSG:32616
finished kde computing.
```

<a href=""><img src="./imgs/5_12.jpg" height="auto" width="auto" title="caDesign"></a>

* The second method calculates the kernel density estimation and saves it as a continuous raster value.

```python
import util 
import rasterio
start_time=util.start_time()
raster_path=r'./data/geoData/amenity_G_kde.tif'
cellSize=500
scale=10**10 
amenity_G_kde=util.pts_geoDF2raster(amenity_poi_copy,raster_path,cellSize,scale)
util.duration(start_time)
```

```
start time: 2020-07-12 16:36:33.980935
Start calculating kde...
Finish calculating kde!
conversion complete!
end time: 2020-07-12 16:41:04.800500
Total time spend:4.50 minutes
```

```python
from rasterio.plot import show
import matplotlib.pyplot as plt
amenity_kde=rasterio.open(raster_path)
plt.figure(figsize=(10,15))
show((amenity_kde,1),cmap='Greens') 
plt.show()
```

<a href=""><img src="./imgs/5_13.png" height="auto" width="auto" title="caDesign"></a>

### 1.3 key point
#### 1.3.1 data processing technique

* Summary of Numpy processing techniques

1. Establishment of data

-Create an arithmetic sequence,
`x=np.linspace(stats.norm.ppf(0.001,loc=0,scale=1),stats.norm.ppf(0.999,loc=0,scale=1), 100)` # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)

2. Data organization

-Sort an array 
`rVar_sort=np.sort(random_variates)` #numpy.sort(a, axis=-1, kind=None, order=None)

-Stack arrays vertically in order
`values=np.vstack([pts_geoDF.geometry.x, pts_geoDF.geometry.y]) `

-Update array structure（shape）
`Z=np.reshape(kernel(positions).T, X.shape)` #numpy.flip(m, axis=None)

-Flip the array
`np.flip(Z,0)`

* GDAL library, processing geospatial data, including raster, vector, and coordinate projection system, etc.

* Rasterio library, simple raster data processing library

#### 1.3.2 The newly created function tool
* function - Convert point data in .shp format to .tif raster, `pts2raster(pts_shp,raster_path,cellSize,field_name=False)`

* function -  Convert point data in GeoDaraFrame format to raster data, `pts_geoDF2raster(pts_geoDF,raster_path,cellSize,scale)`

#### 1.3.3 The python libraries that are being imported

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math
import pandas as pd
import plotly.express as px
import geopandas as gpd
from pyproj import CRS
import os
from osgeo import gdal, ogr,osr
import rasterio
from rasterio.plot import show
```