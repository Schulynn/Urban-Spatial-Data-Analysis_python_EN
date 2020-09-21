> Created on Sat Aug  8 19/52/37 2020  @author: Richie Bao-caDesign (cadesign.cn)

## 1. Remote sensing image interpretation (based on NDVI), the establishment of sampling tool(GUI_tkinter), confusion matrix
If the only analysis of the urban green space(which can be further subdivided cultivated land and forest land), bare land, and water, nothing to do with more detailed land cover classification, such as thickets, grassland, bare land, residential land, garden land, cultivated land, rivers, lakes, and so on, can use of Landsat series of remote sensing image to extract by NDVI, NDWI, and NDBI. The use of platform-tools like eCognition is recommended for sophisticated classification. The NDVI based remote sensing image interpretation was first to read Landsat's images with different seasons, which reflected the different seasons of green space. The cropping boundary can be completed in QGIS according to the research purpose scope; -> Then the NDVI in different seasons is calculated; -> Then, the interactive Plolyly was used to analyze the value range of NDVI, judge the range of different land cover threshold, and interpret the image. -> If needed to judge the accuracy of interpretation is given sampling, namely random extraction point real land cover types; this process is a manual process, and the embedded python library [tkinter](https://docs.python.org/3/library/tkinter.html), which build Graphical User Interface(GUI) can easily help us to establish interactive operation platform quickly, to complete the sampling work; -> Finally, the confusion matrix and percentage accuracy are calculated to judge the interpretation accuracy.

### 1.1 Image data processing
Some commonly used tools are usually built as functions to facilitate image processing, such as the following image crop function `raster_clip`. The most important image processing aspect is the coordinate projection system; Landsat images usually contain corresponding projection systems, such as UTM, DATUM: WGS84, UTM_ZONE16. Therefore, the best choice is to unify the coordinate projection system directly, and conversion to other projection systems is not recommended. For other geographic files, raster data usually requires conversion projections, meaning that it has its projection. For vector geographic files .shp, they are generally kept as WGS84, that is, EPSG:4326, and usually do not contain projection, which is convenient for different coordinate projection system platforms and data format types transit. Therefore, when the cutting boundary is established under the QGIS platform, the coordinate system is only maintained as WGS84, and the projection is not configured.  In the defined cropping function, it is defined according to the projection system of the Landsat image read to facilitate data processing.


Image data are generally very large, so image data processing, especially the high-resolution image, takes time. Therefore, it is necessary to save the processed data immediately to the hard disk space, and directly read it from the hard disk when needed, to avoid spending time on calculation again.


```python
import util
import os
workspace=r"F:\data_02_Chicago\9_landsat\data_processing"
Landsat_fp={
        "w_180310":r"F:\data_02_Chicago\9_landsat\LC08_L1TP_023031_20180310_20180320_01_T1", #Winter
        "s_190820":r"F:\data_02_Chicago\9_landsat\LC08_L1TP_023031_20190804_20190820_01_T1", #Summer
        "a_191018":r"F:\data_02_Chicago\9_landsat\LC08_L1TP_023031_20191007_20191018_01_T1" #Autumn
        }

w_180310_band_fp_dic,w_180310_Landsat_para=util.LandsatMTL_info(Landsat_fp["w_180310"]) #LandsatMTL_info(fp) function, described in the Landsat remote sensing image processing section, is placed in the util.py file and then called.
s_190820_band_fp_dic,s_190820_Landsat_para=util.LandsatMTL_info(Landsat_fp["s_190820"])
a_191018_band_fp_dic,a_191018_Landsat_para=util.LandsatMTL_info(Landsat_fp["a_191018"])

print("w_180310-MAP_PROJECTION:%s,DATUM:%s,UTM_ZONE%s"%(w_180310_Landsat_para['MAP_PROJECTION'],w_180310_Landsat_para['DATUM'],w_180310_Landsat_para['UTM_ZONE']))
print("s_190820-MAP_PROJECTION:%s,DATUM:%s,UTM_ZONE%s"%(s_190820_Landsat_para['MAP_PROJECTION'],s_190820_Landsat_para['DATUM'],s_190820_Landsat_para['UTM_ZONE']))
print("a_191018-MAP_PROJECTION:%s,DATUM:%s,UTM_ZONE%s"%(a_191018_Landsat_para['MAP_PROJECTION'],a_191018_Landsat_para['DATUM'],a_191018_Landsat_para['UTM_ZONE']))

def raster_clip(raster_fp,clip_boundary_fp,save_path):
    import earthpy.spatial as es
    import geopandas as gpd
    from pyproj import CRS
    import rasterio as rio
    '''
    function - Given cutting boundaries, batch cropping raster data
    
    Paras:
    raster_fp - Raster data file paths(.tif) to be trimmed with the same coordinate projection system.
    clip_boundary - The boundary for cropping(.shp, WGS84, no projection)
    
    return:
    rasterClipped_pathList - The list of file paths after cropping
    '''
    clip_bound=gpd.read_file(clip_boundary_fp)
    with rio.open(raster_fp[0]) as raster_crs:
        raster_profile=raster_crs.profile
        clip_bound_proj=clip_bound.to_crs(raster_profile["crs"])
    
    rasterClipped_pathList=es.crop_all(raster_fp, save_path, clip_bound_proj, overwrite=True) #Perform crop on all bands
    print("finished clipping.")
    return rasterClipped_pathList
    
clip_boundary_fp=r".\data\geoData\LandsatChicago_boundary.shp"
save_path=r"F:\data_02_Chicago\9_landsat\data_processing\s_190820"    
s_190820_clipped_fp=raster_clip(list(s_190820_band_fp_dic.values()),clip_boundary_fp,save_path)    
```

    w_180310-MAP_PROJECTION:UTM,DATUM:WGS84,UTM_ZONE16
    s_190820-MAP_PROJECTION:UTM,DATUM:WGS84,UTM_ZONE16
    a_191018-MAP_PROJECTION:UTM,DATUM:WGS84,UTM_ZONE16
    finished clipping.
    

The winter and autumn images are calculated directly by the defined cropping function.



```python
save_path=r"F:\data_02_Chicago\9_landsat\data_processing\w_180310"    
w_180310_clipped_fp=raster_clip(list(w_180310_band_fp_dic.values()),clip_boundary_fp,save_path) 

save_path=r"F:\data_02_Chicago\9_landsat\data_processing\a_191018"    
a_191018_clipped_fp=raster_clip(list(a_191018_band_fp_dic.values()),clip_boundary_fp,save_path) 
```

    finished clipping.
    finished clipping.
    

Landsat images downloaded from the U.S. Geographical Survey, each band is a separate file.  The .stack method provided by the Earthpy library is used to place all bands under a single file in the form of an array, for easy data processing to avoid reading a single file at a time.



```python
import earthpy.spatial as es
w_180310_array, w_180310_raster_prof=es.stack(w_180310_clipped_fp[:7], out_path=os.path.join(workspace,r"w_180310_stack.tif"))
print("finished stacking_1...")

s_190820_array, s_190820_raster_prof=es.stack(s_190820_clipped_fp[:7], out_path=os.path.join(workspace,r"s_190820_stack.tif"))
print("finished stacking_2...")

a_191018_array, a_191018_raster_prof=es.stack(a_191018_clipped_fp[:7], out_path=os.path.join(workspace,r"a_191018_stack.tif"))
print("finished stacking_3...")
```

    finished stacking_1...
    finished stacking_2...
    finished stacking_3...
    


Only by displaying the data can we better judge whether the result of geospatial data processing is correct. Set up `bands_show` image band display function to view the image. The input parameter band_num can determine the band displayed in the synthesis.



```python
def bands_show(img_stack_list,band_num):
    import matplotlib.pyplot as plt
    from rasterio.plot import plotting_extent
    import earthpy.plot as ep
    '''
    function - Specify the band and display multiple remote sensing images at the same time.
    
    Paras:
    img_stack_list - Image list
    band_num - Display the layer
    '''
    
    def variable_name(var):
        '''
        function - Converts a variable name to a string
        
        Paras:
        var - The variable name
        '''
        return [tpl[0] for tpl in filter(lambda x: var is x[1], globals().items())][0]

    plt.rcParams.update({'font.size': 12})
    img_num=len(img_stack_list)
    fig, axs = plt.subplots(1,img_num,figsize=(12*img_num, 12))
    i=0
    for img in img_stack_list:
        ep.plot_rgb(
                    img,
                    rgb=band_num,
                    stretch=True,
                    str_clip=0.5,
                    title="%s"%variable_name(img),
                    ax=axs[i]
                )
        i+=1
    plt.show()
img_stack_list=[w_180310_array,s_190820_array,a_191018_array]
band_num=[3,2,1]
bands_show(img_stack_list,band_num)
```


<a href=""><img src="./imgs/11_07.png" height="auto" width="auto" title="caDesign"></a>



From the synthetic band images directly displayed above, the display effect is dark, which is not conducive to observing details. The exposure method in the [scikit-image](https://scikit-image.org/) image processing library allows us to stretch the image, lighten, and enhance the contract. The processing needs to be done on a band by band basis and then merged.


```python
def image_exposure(img_bands,percentile=(2,98)):
    from skimage import exposure
    import numpy as np
    '''
    function - contract stretching images
    
    Paras:
    img_bands - landsat The band processed by stack
    percentile - Percentile
    
    return:
    img_bands_exposure - Returns the stretched image
    '''
    bands_temp=[]
    for band in img_bands:
        p2, p98=np.percentile(band, (2, 98))
        bands_temp.append(exposure.rescale_intensity(band, in_range=(p2,p98)))
    img_bands_exposure=np.concatenate([np.expand_dims(b,axis=0) for b in bands_temp],axis=0)
    print("finished exposure.")
    return img_bands_exposure

w_180310_exposure=image_exposure(w_180310_array)
s_190820_exposure=image_exposure(s_190820_array)
a_191018_exposure=image_exposure(a_191018_array)
```

    finished exposure.
    finished exposure.
    finished exposure.
    

Compared with the original display effect, it can be seen that the processed remote sensing image has obvious improvement.



```python
img_stack_list=[w_180310_exposure,s_190820_exposure,a_191018_exposure]
band_num=[3,2,1]
bands_show(img_stack_list,band_num)
```


<a href=""><img src="./imgs/11_08.png" height="auto" width="auto" title="caDesign"></a>



### 1.2 Computing NDVI, interactive image, and interpretation
The NDVI calculation function `NDVI(RED_band, NIR_band)` defined in the description of the image band calculation index is called to calculate NDVI(Normalized Vegetation Index). Simultaneously, by examining the maximum and minimum values of the calculation results, namely the interval of the values, it can be seen s_190820_NDVI data(summer) has obvious errors. By viewing the above images, it can also be found that the download Landsat images in summer have a lot of clouds in the urban area, which may be the causes of the data anomaly.


```python
import util
import rasterio as rio
import os
workspace=r"F:\data_02_Chicago\9_landsat\data_processing"
w_180310=rio.open(os.path.join(workspace,r"w_180310_stack.tif"))
s_190820=rio.open(os.path.join(workspace,r"s_190820_stack.tif"))
a_191018=rio.open(os.path.join(workspace,r"a_191018_stack.tif"))

w_180310_NDVI=util.NDVI(w_180310.read(4),w_180310.read(5))
s_190820_NDVI=util.NDVI(s_190820.read(4),s_190820.read(5))
a_191018_NDVI=util.NDVI(a_191018.read(4),a_191018.read(5))
```

    NDVI_min:0.000000,max:15.654902
    NDVI_min:-9999.000000,max:8483.000000
    NDVI_min:0.000000,max:15.768559
    

Because NDVI data is a dimensional array, we can directly use the `is_outlier(data,threshold=3.5)` function defined in the outlier processing section to process outliers and print NDVI image to see the processing results. NDVI's recognition of green vegetation can be seen more clearly, which can better distinguish the water body from bare land.


```python
import matplotlib.pyplot as plt
import earthpy.plot as ep
fig, axs=plt.subplots(1,3,figsize=(30, 12))
ep.plot_bands(w_180310_NDVI, cbar=False, title="w_180310_NDVI", ax=axs[0],cmap='flag')

s_190820_NDVI=util.NDVI(s_190820.read(4),s_190820.read(1)) #The outlier is replaced in situ. Therefore, if the outlier detection is run multiple times, NDVI must be recalculated to keep the original data unchanged.
is_outlier_bool,_=util.is_outlier(s_190820_NDVI,threshold=3)
s_190820_NDVI[is_outlier_bool]=0 #In situ to replace
ep.plot_bands(s_190820_NDVI, cbar=False, title="s_190820_NDVI", ax=axs[1],cmap='terrain')

ep.plot_bands(a_191018_NDVI, cbar=False, title="a_191018_NDVI", ax=axs[2],cmap='flag')
plt.show()
```

    NDVI_min:0.000000,max:64105.000000
    


<a href=""><img src="./imgs/11_09.png" height="auto" width="auto" title="caDesign"></a>


Directly processed in the form of (4900, 4604) NDVI array, containing $4900 \times 4604$ multiple data, if the computer hardware conditions allow, to the extent that the computation time is affordable, can not be compressed data. Otherwise, without affecting the final calculation results on the analysis, the image can be compressed appropriately. The method of 'rescale' in [scikit-image](https://scikit-image.org/) image processing library can be directly used, 'resize', 'downscale_local_mean' are also called in, and you can view their functionalities for yourself. The general method of calculation is to take the local mean.

The NDVI obtained from summer image calculation is printed to display the color images according to the threshold value, which is convenient for viewing the area range of different threshold intervals and determining the threshold value to interpret the landcover. 


```python
import plotly.express as px
from skimage.transform import rescale, resize, downscale_local_mean
s_190820_NDVI_rescaled=rescale(s_190820_NDVI, 0.1, anti_aliasing=False)
fig=px.imshow(img=s_190820_NDVI_rescaled,zmin=s_190820_NDVI_rescaled.min(),zmax=s_190820_NDVI_rescaled.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.speed)
fig.show()
```

<a href=""><img src="./imgs/11_10.png" height="auto" width="800" title="caDesign"></a>

The NDVI value calculated is a dimensionless array of dimensions 1, usually floating-point decimal values. Using chart libraries such as Matplotlib, Plotly, Seaborn, and Bokeh, RGB, RGBA, hexadecimal, and floating-point data formats for colors are commonly used. For the computed NDVI, the function 'data_division' is defined to convert it to RGB's color format according to the classification threshold to facilitate image printing. Because NDVI itself is not color data, a percentile is calculated based on the classification threshold. Then, based on its percentile, the .digitize method is used to return the value's corresponding interval index(integer). The number of intervals, that is, the number of unique indexes, defines different random integers between 0 and 255 (each value corresponds to three random integer values), representing the color.

```python
from skimage.transform import rescale, resize, downscale_local_mean
def data_division(data,division,right=True):
    import numpy as np
    import pandas as pd
    '''
    function - Divide the data by a given percentile and give a fixed value, integer, or RGB color value
    
    Paras:
    data - Numpy array to partition
    division - Percentile list
    
    return：
    data_digitize - Return integer value
    data_rgb - Returns RGB, color value
    '''
    percentile=np.percentile(data,np.array(division))
    data_digitize=np.digitize(data,percentile,right)
    
    unique_digitize=np.unique(data_digitize)
    random_color_dict=[{k:np.random.randint(low=0,high=255,size=1)for k in unique_digitize} for i in range(3)]
    data_color=[pd.DataFrame(data_digitize).replace(random_color_dict[i]).to_numpy() for i in range(3)]
    data_rgb=np.concatenate([np.expand_dims(i,axis=-1) for i in data_color],axis=-1)
    
    return data_digitize,data_rgb

w_180310_NDVI_rescaled=rescale(w_180310_NDVI, 0.1, anti_aliasing=False)
s_190820_NDVI_rescaled=rescale(s_190820_NDVI, 0.1, anti_aliasing=False)
a_191018_NDVI_rescaled=rescale(a_191018_NDVI, 0.1, anti_aliasing=False)
print("finished rescale .")
```

    finished rescale .


```python
division=[0,35,85]
_,s_190820_NDVI_RGB=data_division(s_190820_NDVI_rescaled,division)
fig=px.imshow(img=s_190820_NDVI_RGB,zmin=s_190820_NDVI_RGB.min(),zmax=s_190820_NDVI_RGB.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.speed)
fig.show()
```

<a href=""><img src="./imgs/11_06.png" height="auto" width="800" title="caDesign"></a>


The data distribution of NDVI can be viewed through the histogram(frequency distribution) to assist interpretation and determine the threshold. Because of the different number of ground objects, certain points of transition, or rupture, may represent different features.

```python
fig, axs=plt.subplots(1,3,figsize=(20, 6))

count, bins, ignored = axs[0].hist(w_180310_NDVI.flatten(), bins=100,density=True) 
count, bins, ignored = axs[1].hist(s_190820_NDVI.flatten(), bins=100,density=True) 
count, bins, ignored = axs[2].hist(a_191018_NDVI.flatten(), bins=100,density=True) 
plt.show()
```

<a href=""><img src="./imgs/11_11.png" height="auto" width="auto" title="caDesign"></a>

Although the above code can be printed to display NDVI images according to the threshold, the interactive operation cannot be performed; that is, through constant adjustment of the threshold interval, the threshold image can be immediately viewed to determine the threshold value suitable for the classification of ground objects. The Plotly diagram tool provides a simple interactive tool that can be interactively performed by adding a slider window tool(widgets.IntSlier) and a drop-down bar(widgets.Dropdown). Here, three sliders are defined to define the threshold interval; a drop-down bar defines optional NDVI data for three different seasons, calls the  `data_division` function defined above, and configures the color to display the image according to the threshold interval. By constantly adjusting the threshold value and comparing it with the true-color images of band 3,2 and 1, it is judged that the threshold interval of landcover classification in three seasons is [0,35,85]. Note that NDVI classification thresholds may vary from image to image.

```python
def percentile_slider(season_dic):
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from ipywidgets import widgets
    from IPython.display import display
    
    '''
    function - Multiple raster data, given percentile, observe changes.
    
    Paras:
    season_dic -  Multiple raster dictionaries
    '''
    
    p_1_slider=widgets.IntSlider(min=0, max=100, value=10, step=1, description="percentile_1")
    p_2_slider=widgets.IntSlider(min=0, max=100, value=30, step=1, description="percentile_2")
    p_3_slider=widgets.IntSlider(min=0, max=100, value=50, step=1, description="percentile_3")

    
    season_keys=list(season_dic.keys())

    season=widgets.Dropdown(
        description='season',
        value=season_keys[0],
        options=season_keys
    )

    season_val=season_dic[season_keys[0]]
    _,img=data_division(season_val,division=[10,30,50],right=True)
    trace1=go.Image(z=img)

    g=go.FigureWidget(data=[trace1,],
                      layout=go.Layout(
                      title=dict(
                      text='NDVI interpretation'
                            ),
                      width=800,
                      height=800
                        ))

    def validate():
        if season.value in season_keys:
            return True
        else:
            return False

    def response(change):
        if validate():
            division=[p_1_slider.value,p_2_slider.value,p_3_slider.value]
            _,img_=data_division(season_dic[season.value],division,right=True)
            with g.batch_update():
                g.data[0].z=img_
    p_1_slider.observe(response, names="value")            
    p_2_slider.observe(response, names="value")    
    p_3_slider.observe(response, names="value")    
    season.observe(response, names="value")    

    container=widgets.HBox([p_1_slider,p_2_slider,p_3_slider,season])
    box=widgets.VBox([container,
                  g])
    display(box)
    
season_dic={"w_180310":w_180310_NDVI_rescaled,"s_190820":s_190820_NDVI_rescaled,"a_191018":a_191018_NDVI_rescaled}    
percentile_slider(season_dic)  
```

<a href=""><img src="./imgs/11_04.png" height="auto" width="auto" title="caDesign"></a>

After determining the threshold interval of [0,35,85], all seasons' NDVI land classification can be calculated.

```python
division=[0,35,85]
w_interpretation,_=data_division(w_180310_NDVI_rescaled,division,right=True)
s_interpretation,_=data_division(s_190820_NDVI_rescaled,division,right=True)
a_interpretation,_=data_division(a_191018_NDVI_rescaled,division,right=True)
```

After obtaining the interpretation results of NDVI in three different seasons, because most of the farmland was no planted in winter, some of the farmland was harvested in autumn. Most of the farmland grew in summer, and deciduous trees flourish in summer. Evergreen trees remain green throughout the year, landcover types, such as farmland, evergreen, deciduous, bare land, and water body, can be interpreted. Here, as long as each season is green, it is divided into green(.logical_or) with a value of 2; However, the threshold value of lakes in water bodies is well distinguished, while that of rivers is not very clear. Therefore, it is determined that only water bodies are water bodies(.logical_and), with the value 3; The rest is bare, with values 0 and 1. 


```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

fig=make_subplots(rows=1, cols=4,shared_xaxes=False,subplot_titles=('w_interpretation',  's_interpretation', 'a_interpretation', 'green_water_bareLand'))

fig1=px.imshow(img=np.flip(w_interpretation,0),zmin=w_interpretation.min(),zmax=w_interpretation.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.deep,title="winter")
fig2=px.imshow(img=np.flip(s_interpretation,0),zmin=s_interpretation.min(),zmax=s_interpretation.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.haline)
fig3=px.imshow(img=np.flip(a_interpretation,0),zmin=a_interpretation.min(),zmax=a_interpretation.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.haline)

green=np.logical_or(w_interpretation==2,s_interpretation==2,a_interpretation==2) #As long as it is a greenfield(value 2), it is 2.
water=np.logical_and(w_interpretation==3,s_interpretation==3,a_interpretation==3) #As long as it is a water body(value 3), it is 3.
green_v=np.where(green==True,2,0)
water_v=np.where(water==True,3,0)
green_water_bareLand=green_v+water_v
fig4=px.imshow(img=np.flip(green_water_bareLand,0),zmin=green_water_bareLand.min(),zmax=green_water_bareLand.max(),width=800,height=800,color_continuous_scale=px.colors.sequential.haline)

trace1=fig1['data'][0]
trace2=fig2['data'][0]
trace3=fig3['data'][0]
trace4=fig4['data'][0]

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=1, col=2)
fig.add_trace(trace3, row=1, col=3)
fig.add_trace(trace4, row=1, col=4)

fig.update_layout(height=500, width=1600, title_text="interpretation")      
fig.show()
```

<a href=""><img src="./imgs/11_12.png" height="auto" width="auto" title="caDesign"></a>

### 1.3 Establishment of sampling interactive operation platform and precision calculation
#### 1.3.1  Use Tkinter to set up sampling interactive operation platform
For sampling, the data format of .shp points can be established with QGIS and other platforms to randomly generate a certain number of points. The landcover type of each point can be determined visually or do it by hand. The sampled data is then read under python for subsequent precision analysis. This method requires data conversion on different platforms, and the operation is a little tedious. Here, let's build our own interactive GUI sampling tool through Tkinter. The autumn image was selected as a base map reference, and the image is saved separately for easy invocation.




```python
import matplotlib.pyplot as plt
import earthpy.plot as ep
from skimage.transform import rescale, resize, downscale_local_mean
import os
a_191018_exposure_rescaled=np.concatenate([np.expand_dims(rescale(b, 0.1, anti_aliasing=False),axis=0) for b in a_191018_exposure],axis=0)
fig, ax= plt.subplots(figsize=(12, 12))
ep.plot_rgb(
            a_191018_exposure_rescaled,
            rgb=[3,2,1],
            stretch=True,
            #extent=extent,
            str_clip=0.5,
            title="a_191018_exposure_rescaled",
            ax=ax
        )
plt.show()
print("a_191018_exposure_rescaled shape:",a_191018_exposure_rescaled.shape)
save_path=r"C:\Users\richi\omen-richiebao\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\data"
np.save(os.path.join(save_path,'a_191018_exposure_rescaled.npy'),a_191018_exposure_rescaled)
```

<a href=""><img src="./imgs/11_13.png" height="auto" width="auto" title="caDesign"></a>

* Sample size



Using the sampling method to estimate the accuracy, the smaller the sample, the larger the overall estimator error. The minimum number of samples a set of samples should contain should be calculated within the application's permissible error range. Its formula is ：$n= \frac{pq z^{2} }{ d^{2} }$, where $p$ and $q$ are the percentages of correct and incorrect interpretation respectively, or expressed as $p(1-p)$; z is the bilateral critical value corresponding to the confidence level; $d$ is the error allowed range. The interpretation accuracy is assumed to be 90%, and the confidence level is 95%. And the critical bilateral value is 1.95996, when the confidence level is 95% by using the norm tool of scipy.stats. The sample number can be calculated as about 138 when substituted into the formula.


```python
from scipy.stats import norm
import math
val=norm.ppf((1+0.95)/2)
n=val**2*0.9*0.1/0.05**2
print("The sample size is:",n)
```

    The sample size is: 138.2925175449885

The code structure of the sample interactive platform(shown in Markdown)


```mermaid
classDiagram

The_functionalities_contained_in_the_GUI --> 5_Image_data_processing : a.Data reading and preprocessing
5_Image_data_processing : 1_ Array image(numpy-array)
5_Image_data_processing : 2_Image data(.jpg,.png...)
5_Image_data_processing --> class_main_window : a.

The_functionalities_contained_in_the_GUI : b_Display_background_image_zoomable
The_functionalities_contained_in_the_GUI : c_Select_the_category_and_sample_in_the_image
The_functionalities_contained_in_the_GUI : d_Displays_the_current_sample_classification_status
The_functionalities_contained_in_the_GUI : e. Calculate_the_sampling_position_coordinates
The_functionalities_contained_in_the_GUI : a. 5_Image_data_processing()
The_functionalities_contained_in_the_GUI : -. Python_console_information_print()

The_functionalities_contained_in_the_GUI --> 1_Display_background_image_zoomable : b.Image display
1_Display_background_image_zoomable --> Display_background_image_zoomable : b.
1_Display_background_image_zoomable : 1_Display_an_image_on_the_canvas
1_Display_background_image_zoomable : 2. Add_scroll_bars
1_Display_background_image_zoomable : 3. Mouse_wheel_zoom

class_CanvasImage o-- class_main_window
class_CanvasImage --o class_AutoScrollbar
class_CanvasImage : A_handler_that_basically_includes_all_functions
class_CanvasImage : def __init__()
class_CanvasImage : def compute_samplePosition()
class_CanvasImage : def click_xy_collection()
class_CanvasImage : def __scroll_x()
class_CanvasImage : def _scroll_y()
class_CanvasImage : def __move_from()
class_CanvasImage : def __move_to()
class_CanvasImage : def __wheel()
class_CanvasImage : def show_image()

class_CanvasImage <|-- Display_background_image_zoomable: b.
Display_background_image_zoomable : Scroll_bar_drag
Display_background_image_zoomable : Click_and_drag
Display_background_image_zoomable : Mouse_wheel

Display_background_image_zoomable : 
Display_background_image_zoomable :  def __scroll_x()
Display_background_image_zoomable : def _scroll_y()
Display_background_image_zoomable : def __move_from()
Display_background_image_zoomable :  def __move_to()
Display_background_image_zoomable : def __wheel()
Display_background_image_zoomable : def show_image()

class_main_window *-- Outside_calls
class_main_window : The_main_program
class_main_window : Image_reading_and_preprocessing
class_main_window : def __init__()
class_main_window : def landsat_stack_array2img()

Outside_calls : Define_the_workspace-workspace
Outside_calls : Instantiate_the_call-app=main_window
Outside_calls : Save_sampled_data-pickle

class_AutoScrollbar : The_scroll_bar_defaults_to_a_hidden_configuration
class_AutoScrollbar : Exception_handling
class_AutoScrollbar : def set()
class_AutoScrollbar : def pack()
class_AutoScrollbar : def place)
class_AutoScrollbar -- Display_background_image_zoomable : b.

The_functionalities_contained_in_the_GUI --> 2_Select_classification_to_sample_in_the_image : c.Classification_of_the_sample
2_Select_classification_to_sample_in_the_image : 1. Layout_3_radio_buttons-Radiobutton
2_Select_classification_to_sample_in_the_image --> Left_click_on_sample : c.

The_functionalities_contained_in_the_GUI --> 3_Displays_the_current_sample_classification_status : d.Classification_state
3_Displays_the_current_sample_classification_status : 1. Layout_1_label-Label

2_Select_classification_to_sample_in_the_image --> 3_Displays_the_current_sample_classification_status : c-d

class_CanvasImage <|-- Left_click_on_sample : c.
Left_click_on_sample : def click_xy_collection()
Left_click_on_sample : Store_click_coordinates_by_category

class_CanvasImage <|-- Initialize : comprehensive
Initialize : All_processing_is_summarized_here
Initialize : def __init__()
3_Displays_the_current_sample_classification_status --> Initialize : d.
2_Select_classification_to_sample_in_the_image --> Initialize : c.
1_Display_background_image_zoomable --> Initialize : b.
class_main_window --> Initialize : a.

The_functionalities_contained_in_the_GUI --> 4_Calculate_the_sampling_position_coordinates : e.Sampling_coordinate_transformation
4_Calculate_the_sampling_position_coordinates : 1. Restores_the_sampling_point_coordinates_after_the_canvas_has_been_moved_and_scaled

4_Calculate_the_sampling_position_coordinates --> Coordinate_transformation : e.
Coordinate_transformation : def compute_samplePosition()
class_CanvasImage <|-- Coordinate_transformation : e.
4_Calculate_the_sampling_position_coordinates --> Initialize : e.

```



To complete a comprehensive code that can handle one or more tasks, it is generally necessary to combine classes to comb the code structure. If you only rely on functions, you can still complete the task, but the code structure is loose, not convenient for code writing and code viewing. Before starting a task, it is a good idea to have a clear idea of what you want to do. For example, a sampling GUI tool needs to do the following:

1. Image data processing
2. Display the background image, which can be scaled
3. Select the classification and sample the images
4. Display the current sampling classification staus
5. Calculate the coordinates of sampling position
6. Python console information printing


For example, you may want to implement a sampling tool initially for a small functional development, but some details may not be considered. After you start the task, you can make adjustments as the code is written further. However, suppose you start with a tool that is particularly complex in function. In that case, you will improve your code writing efficiency by considering possible problems as clearly as possible, especially the implementation of details and the comprehensive structure, and avoid the code adjustment or even rewriting caused by poor consideration. Of course, even with the clarity of thought, problems will continue to crop up, especially for novices, because the code itself is a process of constant debugging.

The writing code process requires constant debugging, and reading the code is about figuring out the structure of the code, particularly the sequence of function implementation and the correlation between the data. The structure diagram of the above code can well help us identify the structure of the code. If there is no such structure diagram, we need to run the program step by step from the beginning and print() the variable data that needs to be viewed to deduce the whole code process. First, determine the "GUI contains functionality", from the function to be implemented, clear the context. According to the implementation order, identified by a,b,c,d,e, you can follow the alphabetical identification order to see the classes and functions. This diagram does not show the specific code lines, but the sequence gives you a sense of the entire code structure. The corresponding line of code can be easily found according to the corresponding function given.


There are two key points to note in the development of this handy tool. One is image scaling; The other one is the transformation of the sample point coordinates. For the first problem, the code given by [stack overflow-tkinter canvas zoom+ move/pan](https://stackoverflow.com/questions/41656176/tkinter-canvas-zoom-move-pan) is transferred. Because the original code is to read the image file directly, but, the remote sensing image band file with an array structure needs to be read here, the image processing code part needs to be added, and the source code needs to be modified to adapt to the added part. The core of the source code for image zooming does not need to be adjusted, including scroll bar configuration, left-drag configuration, and mouse scroll zooming, while the core is the image display function `show_image(self, event=None)`. Because the image movement and scaling cause the scaling of the canvas, the scroll bar(_scroll_x/__scroll_y), and the mouse drag(_move_from/__move_to) cause the movement, and the mouse wheel (__wheel) causes the scaling of the canvas.  Scaling needs to control the image size. When the image is reduced to a certain extent, the image will no longer be shrunk, controlled by the image scaling factor (_wheel) parameter with the initial value of 1.0. The delta parameter controls the scale of each scroll wheel. We can try to modify different parameter values to observe the change of the image. Because of the movement and scaling of the canvas, the image display needs to be adjusted accordingly, read the current image position(bbox1), the canvas visual area(bbox2), calculate the scroll bar area(bbox), and determine whether the image is partially or completely located in the canvas area. If it is part, calculate the tiled coordinates ($ x_{1} , y_{1} , x_{2} , y_{2}$) of the image visible region, and crop the image, otherwise the entire image. The image is scaled with the canvas `self.canvas.scale('all', x, y, scale, scale)`, that is, scaled the canvas and all objects on the canvas. The best way to understand this part is to run the code, print the relevant variable, and see how the data changes.

The second problem of the sampling point transformation arises because the image is zoomed in and out and moved. The basic idea of sampling is to determine the category to be sampled, controlled by three radio buttons, click the mouse to trigger the `click_xy_collection` function, draw points(actually circles), and save points by category to the dictionary(xy). And the coordinates of the sample point is determined by the current canvas; the canvas is move and scale, the coordinates of the sample point changes over the canvas, then how to obtain the coordinates of the sample point corresponding to the actual image size(An image is an array, represented by rows and columns, so the coordinate of a pixel can be represented as $( x_{i}, y_{j} )$. Note that the image $x_{i}$ is the column, $y_{j}$ is the row; it is the opposite of Numpy array.  For a Numpy array,  $x_{i}$ is the row, and $y_{j}$ is the column)? The image's scaling is described in the section "A code representation of the linear algebra basis", where can be returned to the original coordinates by a linear transformation directly. The scaling factor, namely 'scale' parameter, needs to be obtained to achieve the linear transformation. Two points can also be automatically generated when the canvas is not moved or scaled to determine this parameter. Their coordinates are calculated; that is, the actual coordinates are taken as references. These two points are then scaled along with the rest of the sample points, and the coordinates are transformed.  In the new canvas space, get the current coordinates and calculate the distance ratio before and after these two points changes: the scaling scale, which is consistent with the scale parameter.  Therefore, 1/scale can be used as the scaling factor of a linear transformation to return the state before scaling. The function `compute_samplePosition`(text displayed in GUI is 'calculate sampling position') is implemented using point coordinate transformation by clicking the button(button_computePosition). The linear transformation is computed using np.matmul(), that is, the product of two matrices. It is better to save the calculated sample point coordinates according to the given file location for subsequent precision analysis.

> Note that the GUI written by Tkinter cannot run under Jupyter(Lab), need to open, and run under the interpreter such as Spyder. You can create a .py file and copy the following code to it before running it. If you write your own GUO tools based on Tkinter development, you also need to write and debug under the Spyder and other interpreters.

```python
import math,os,random
import warnings
import tkinter as tk
import numpy as np
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.util import img_as_ubyte

class AutoScrollbar(ttk.Scrollbar):
    '''Scrollbars are hidden by default'''
    def set(self,low,high):
        if float(low)<=0 and float(high)>=1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self,low,high)
    
    def pack(self,**kw):
        raise tk.TclError("Cannot use pack with the widget"+self.__class_.__name__)

    def place(self,**kw):
        raise tk.TclError("Cannot use pack with the widget"+self.__class_.__name__)
```


```python
class CanvasImage(ttk.Frame):
    '''Display image,can be scaled'''
    def __init__(self,mainframe,img):
        '''Initialize the Frame'''
        ttk.Frame.__init__(self,master=mainframe)
        self.master.title("pixel sampling of remote sensing image")  
        self.img=img
        self.master.geometry('%dx%d'%self.img.size) 
        self.width, self.height = self.img.size

        #Add horizontal and vertical scroll bars
        hbar=AutoScrollbar(self.master, orient='horizontal')
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar.grid(row=1, column=0,columnspan=4, sticky='we')
        vbar.grid(row=0, column=4, sticky='ns')
        #Create the canvas and bind the scroll bar
        self.canvas = tk.Canvas(self.master, highlightthickness=0, xscrollcommand=hbar.set, yscrollcommand=vbar.set,width=self.width,height=self.height)        
        self.canvas.config(scrollregion=self.canvas.bbox('all')) 
        self.canvas.grid(row=0,column=0,columnspan=4,sticky='nswe')
        self.canvas.update() #Update the canvas
        hbar.configure(command=self.__scroll_x) #Bind the scroll bar to the canvas
        vbar.configure(command=self.__scroll_y)
     
        self.master.rowconfigure(0,weight=1) #Makes the canvas (display image) extensible
        self.master.columnconfigure(0,weight=1)              
        
        #Bind events to the canvas
        self.canvas.bind('<Configure>', lambda event: self.show_image())  #Resize the canvas
        self.canvas.bind('<ButtonPress-1>', self.__move_from) #Original canvas position
        self.canvas.bind('<B1-Motion>', self.__move_to) #Move the canvas to a new position
        self.canvas.bind('<MouseWheel>', self.__wheel) #Zoom on Windows and macOS, not on Linux
        self.canvas.bind('<Button-5>', self.__wheel) #Under Linux, scroll down to zoom
        self.canvas.bind('<Button-4>',   self.__wheel) #Under Linux, scroll up to zoom
        #Handle idle keystrokes, because too many keystrokes can slow down a low-performance computer
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))
        
        self.imscale=1.0 #Image scaling
        self.delta=1.2 #Wheel, canvas scale factor       
        
        #Place the image in a rectangular container with the width and height equal to the size of the image
        
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
      
        self.show_image()     
        
        self.xy={"water":[],"vegetation":[],"bareland":[]}
        self.canvas.bind('<Button-1>',self.click_xy_collection)

        self.xy_rec={"water":[],"vegetation":[],"bareland":[]}
        
        #Configure buttons to select samples and calculate sample positions
        button_frame=tk.Frame(self.master,bg='white', width=5000, height=30, pady=3).grid(row=2,sticky='NW')
        button_computePosition=tk.Button(button_frame,text='calculate sampling position',fg='black',width=25, height=1,command=self.compute_samplePosition).grid(row=2,column=0,sticky='w')
        
        self.info_class=tk.StringVar(value='empty')
        button_green=tk.Radiobutton(button_frame,text="vegetation",variable=self.info_class,value='vegetation').grid(row=2,column=1,sticky='w')
        button_bareland=tk.Radiobutton(button_frame,text="bareland",variable=self.info_class,value='bareland').grid(row=2,column=2,sticky='w')    
        button_water=tk.Radiobutton(button_frame,text="water",variable=self.info_class,value='water').grid(row=2,column=3,sticky='w') 

        self.info=tk.Label(self.master,bg='white',textvariable=self.info_class,fg='black',text='empty',font=('Arial', 12), width=10, height=1).grid(row=0,padx=5,pady=5,sticky='nw')
        self.scale_=1
        
        #Draw a reference point
        self.ref_pts=[self.canvas.create_oval((0,0,1.5,1.5),fill='white'), self.canvas.create_oval((self.width,self.height,self.width-0.5, self.height-0.5),fill='white')] 
        
        self.ref_coordi={'ref_pts':[((self.canvas.coords(i)[2]+self.canvas.coords(i)[0])/2,(self.canvas.coords(i)[3]+self.canvas.coords(i)[1])/2) for i in self.ref_pts]}
        self.sample_coordi_recover={}
        

    def compute_samplePosition(self):
        self.xy_rec.update({'ref_pts':self.ref_pts})
        #print(self.xy_rec)
        sample_coordi={key:[((self.canvas.coords(i)[2]+self.canvas.coords(i)[0])/2,(self.canvas.coords(i)[3]+self.canvas.coords(i)[1])/2) for i in self.xy_rec[key]] for key in self.xy_rec.keys()}
        print("+"*50)
        print("sample coordi:",sample_coordi)
        print("_"*50)
        print(self.ref_coordi)
        print("image size:",self.width, self.height )
        print("_"*50)
        distance=lambda p1,p2:math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
        scale_byDistance=distance(sample_coordi['ref_pts'][0],sample_coordi['ref_pts'][1])/distance(self.ref_coordi['ref_pts'][0],self.ref_coordi['ref_pts'][1])
        print("scale_byDistance:",scale_byDistance)
        print("scale_by_self.scale_:",self.scale_)
        
        #Scale back to the original coordinate system
        
        #x_distance=sample_coordi['ref_pts'][0][0]-self.ref_coordi['ref_pts'][0][0]
        #y_distance=sample_coordi['ref_pts'][0][1]-self.ref_coordi['ref_pts'][0][1]
        f_scale=np.array([[1/scale_byDistance,0],[0,1/scale_byDistance]])
        #f_scale=np.array([[scale_byDistance,0,x_distance],[0,scale_byDistance,y_distance],[0,0,scale_byDistance]])
        #print("x_distance,y_distance:",np.array([x_distance,y_distance]))
        
        sample_coordi_recover={key:np.matmul(np.array(sample_coordi[key]),f_scale) for key in sample_coordi.keys() if sample_coordi[key]!=[]}
        print("sample_coordi_recove",sample_coordi_recover)
        relative_coordi=np.array(sample_coordi_recover['ref_pts'][0])-1.5/2
        sample_coordi_recover={key:sample_coordi_recover[key]-relative_coordi for key in sample_coordi_recover.keys() }
        
        print("sample_coordi_recove",sample_coordi_recover)
        self.sample_coordi_recover=sample_coordi_recover
    
    def click_xy_collection(self,event):
        multiple=self.imscale
        length=1.5*multiple #Adjust the size of the drawn rectangle according to the change of image scale to keep the same size
        
        event2canvas=lambda e,c:(c.canvasx(e.x),c.canvasy(e.y)) 
        cx,cy=event2canvas(event,self.canvas) #cx,cy=event2canvas(event,self.canvas)        
        print(cx,cy)         
        if self.info_class.get()=='vegetation':      
            self.xy["vegetation"].append((cx,cy))  
            rec=self.canvas.create_oval((cx-length,cy-length,cx+length,cy+length),fill='yellow')
            self.xy_rec["vegetation"].append(rec)
        elif self.info_class.get()=='bareland':
            self.xy["bareland"].append((cx,cy))  
            rec=self.canvas.create_oval((cx-length,cy-length,cx+length,cy+length),fill='red')
            self.xy_rec["bareland"].append(rec)        
        elif self.info_class.get()=='water':
            self.xy["water"].append((cx,cy))  
            rec=self.canvas.create_oval((cx-length,cy-length,cx+length,cy+length),fill='aquamarine')
            self.xy_rec["water"].append(rec)    
            
        print("_"*50)
        print("sampling count",{key:len(self.xy_rec[key]) for key in self.xy_rec.keys()})    
        print("total:",sum([len(self.xy_rec[key]) for key in self.xy_rec.keys()]) )
        
    def __scroll_x(self,*args,**kwargs):
        '''Scroll the canvas horizontally and redraw the image'''
        self.canvas.xview(*args,**kwargs)#Rolling horizontal bar
        self.show_image() #Redraw the image
        
    def __scroll_y(self, *args, **kwargs):
        """ Scroll the canvas vertically and redraw the image"""
        self.canvas.yview(*args,**kwargs)  #Vertical scroll
        self.show_image()  #Redraw the image    

    def __move_from(self, event):
        ''' Mouse scroll, the previous coordinate '''
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        ''' Mouse scroll, the next coordinate'''
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  #Redraw the image 

    def __wheel(self, event):
        ''' Mouse wheel zoom '''
        x=self.canvas.canvasx(event.x)
        y=self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # The image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  #If the mouse is inside the image area
        else: return  # Scrolling and zooming is only possible if the mouse is inside the image
        scale=1.0
        # Response to Linux (event.num) or Windows (event.delta) wheel events
        if event.num==5 or event.delta == -120:  # Scroll down
            i=min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # The image size is less than 30 pixels
            self.imscale /= self.delta
            scale/= self.delta
        if event.num==4 or event.delta == 120:  # Scroll up
            i=min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # If 1 pixel is larger than the visible image area
            self.imscale *= self.delta
            scale*= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # Scale all the objects on the canvas
        self.show_image()
        self.scale_=scale*self.scale_

    def show_image(self, event=None):
        ''' Display the image on the canvas'''
        bbox1=self.canvas.bbox(self.container)  #Get image region
        # Remove 1-pixel movement on both sides of 'bbox 1'
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # Gets the visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  #Gets the scroll box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # The entire image is in the visible region
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # The entire image is in the visible region
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # Set the scrolling area
        x1 = max(bbox2[0] - bbox1[0], 0)  # Get the tiled coordinates (x1,y1,x2,y2)
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # Display the image if it is in the visible area
            x = min(int(x2 / self.imscale), self.width)   # Sometimes more than 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...Sometimes not
            image = self.img.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # Set the image as the background
            self.canvas.imagetk=imagetk  # keep an extra reference to prevent garbage-collection

```

```python
class main_window:    
    '''The main window class'''
    def __init__(self,mainframe,rgb_band,img_path=0,landsat_stack=0,):
        '''Read the image'''
        if img_path:
            self.img_path=img_path
            self.__image=Image.open(self.img_path)
        if rgb_band:
            self.rgb_band=rgb_band    
        if type(landsat_stack) is np.ndarray:
            self.landsat_stack=landsat_stack
            self.__image=self.landsat_stack_array2img(self.landsat_stack,self.rgb_band)

        self.MW=CanvasImage(mainframe,self.__image)
 
    def landsat_stack_array2img(self,landsat_stack,rgb_band):
        r,g,b=self.rgb_band
        landsat_stack_rgb=np.dstack((landsat_stack[r],landsat_stack[g],landsat_stack[b]))  #Combine three bands
        landsat_stack_rgb_255=img_as_ubyte(landsat_stack_rgb) #Convert float-point colors to integers from 0 to 255 using methods provided by Skimage
        landsat_image=Image.fromarray(landsat_stack_rgb_255)
        return landsat_image

```

```python
if __name__ == "__main__":
    img_path=r'C:\Users\richi\Pictures\n.png' 
    
    workspace=r"C:\Users\richi\omen-richiebao\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\data"
    img_fp=os.path.join(workspace,'a_191018_exposure_rescaled.npy')
    landsat_stack=np.load(img_fp)        
    
    rgb_band=[3,2,1]          
    mainframe=tk.Tk()
    app=main_window(mainframe, rgb_band=rgb_band,landsat_stack=landsat_stack) #img_path=img_path,landsat_stack=landsat_stack,
    #app=main_window(mainframe, rgb_band=rgb_band,img_path=img_path)
    mainframe.mainloop()
    
    #Save sample points
    import pickle as pkl
    with open(os.path.join(workspace,r'sampling_position.pkl'),'wb') as handle:
        pkl.dump(app.MW.sample_coordi_recover,handle)
```

<a href=""><img src="./imgs/11_03.png" height="auto" width="800" title="caDesign"></a>

#### 1.3.2 Classification accuracy calculation
The confusion matrix, where each row(column) represents an instance prediction of a class, and each column(row) represents an instance of an actual class, is a special contingency table with two dimensions(actual and predicted values), and a set of identical categories in both dimensions. For example, $\begin{bmatrix}&&predicted categories & &\\&&baraland&vegetation&water\\actual categories &bareland &51&2&0\\&vegetation&10&54&0\\ &water&0&2&19 \end{bmatrix} $, it can be interpreted as: a total of 138 samples(sampling points), of which 53 are real bare ground and 2 are misjudged as green space; of which 64 are real green space and 10 were misjudged as bare ground; of which 21 are real water bodies and 2 were misjudged as green space. The confusion matrix is calculated using the 'confusion_matrix' method provided by the Sklearn library.

By calculating the confusion matrix and analyzing it, it is known that there is relatively more green land misjudged as bare land. On the one hand, in the interpretation process, the threshold boundary of 35% can be appropriately enlarged, so that part of green space can be divided into the category of bare land. On the other hand, it may be that in the process of sampling, the sampling points are set to a large extent, covering both green land and bare land, and the final coordinate cannot be determined whether if falls on green land or bare land, resulting in sampling errors. At this point, the diameter of the sampling point can be reduced for more accurate positioning.

In addition to calculating the confusion matrix, the percentage accuracy, that is, the ratio of the correct classification to the total sample number, is also calculated.


```python
import pickle as pkl
import os
import pandas as pd
workspace=r"./data"
with open(os.path.join(workspace,r'sampling_position_138.pkl'),'rb') as handle:
    sampling_position=pkl.load(handle)
sampling_position_int={key:sampling_position[key].astype(int) for key in sampling_position.keys() if key !='ref_pts'}
i=0
sampling_position_int_={}
for key in sampling_position_int.keys():
    for j in sampling_position_int[key]:
        sampling_position_int_.update({i:[j.tolist(),key]})
        i+=1
sampling_position_df=pd.DataFrame.from_dict(sampling_position_int_,columns=['coordi','category'],orient='index') #Convert to Pandas DataFrame data format for ease of data processing
sampling_position_df['interpretation']=[green_water_bareLand[coordi[1]][coordi[0]] for coordi in sampling_position_df.coordi]
interpretation_mapping={1:"bareland",2:"vegetation",3:"water",0:"bareland"}
sampling_position_df.interpretation=sampling_position_df.interpretation.replace(interpretation_mapping)
sampling_position_df.to_pickle(os.path.join(workspace,r'sampling_position_df.pkl'))
```

```python
from sklearn.metrics import confusion_matrix
precision_confusionMatrix=confusion_matrix(sampling_position_df.category, sampling_position_df.interpretation)
precision_percent=np.sum(precision_confusionMatrix.diagonal())/len(sampling_position_df)
print("precision - confusion Matrix:\n",precision_confusionMatrix)     
print("precision - percent:",precision_percent)
```

precision - confusion Matrix:
 [[51  2  0]
 [10 54  0]
 [ 0  2 19]]
precision - percent: 0.8985507246376812

### 1.4 key point
#### 1.4.1 data processing technique

* Work with rasterio, geopandas, earthpy on remote sensing images

* Use Plotly to build an interactive diagram.

* Build interactive GUI sampling tools using Tkinter

#### 1.4.2 The newly created function tool

* function - Given cutting boundaries, batch cropping raster data, `raster_clip(raster_fp,clip_boundary_fp,save_path)`

* function - Specify the band and display multiple remote sensing images at the same time, `bands_show(img_stack_list,band_num)`

* function -Converts a variable name to a string,  `variable_name(var)`

* function - contract stretching images,`image_exposure(img_bands,percentile=(2,98))`

* function - Divide the data by a given percentile and give a fixed value, integer, or RGB color value,`data_division(data,division,right=True)`

* function - Multiple raster data, given percentile, observe changes, `percentile_slider(season_dic)`

* Based on Tkinter, an interactive GUI sampling tool is developed

#### 1.4.3 The python libraries that are being imported

```python 
import os,random

import earthpy.spatial as es
import earthpy.plot as ep

import geopandas as gpd
from pyproj import CRS
import rasterio as rio
from rasterio.plot import plotting_extent

import matplotlib.pyplot as plt

from skimage import exposure
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np
import pandas as pd
from scipy.stats import norm
import math

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets
from IPython.display import display

import warnings
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from skimage.util import img_as_ubyte

import pickle as pkl
from sklearn.metrics import confusion_matrix
```

#### 1.4.4 Reference
-
