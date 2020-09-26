> Created on Mon Dec  2 22/25/18 2019  @author: Richie Bao-caDesign (cadesign.cn)
> __+updated on Fri Aug 14 21/20/01 2020 by Richie Bao

## 1. Point cloud data(Lidar) processing——classified data, DSM, building height extraction, interpolation
Point Cloud is the data obtained using a 3D scanner; of course, the designed 3D model can also be converted into point cloud data. The three-dimensional object is recorded in the form of points, each of which is a three-dimensional coordinate and may contain color information(RGB) or intensity of the object's reflective surface. Intensity information is the echo intensity collected by the laser scanner receiver, which is related to the target surface material, roughness, incidence angle direction, instrument emission energy, and laser wavelength. Point cloud data format is relatively rich, commonly used, including .xyz(.xyzn, .xyzrgb), .las, .ply, .pcd, .pts and so on, also includes some associated format storage types, such as array .numpy(.npu) based on Numpy storage, Matlab array .matlab format,  as well as .txt files based on text storage. Note that although some storage types have different suffix names, the data format may be the same. In geospatial data, .las format is often used. LAS(LASer) format is a file format for exchanging and archiving lidar point cloud data established by the American Society for Photogrammetry and Remote Sensing(ASPRS). It is considered the industry standard for lidar data. LAS format point cloud data includes multiple versions, most recently LAS 1.4(2011.11.14); it is important to note that different point cloud data versions may contain different information. LAS generally includes the classification information identified by the integer value (LAS 1.1 and later versions). Its 1.1-1.4LAS classification is as follows:

|  classification value | classification  |   
|---|---|
|0   | Never classified   |      
|1   | unassigned  |      
|2   | ground  |      
|3  | low vegetation  | 
|4  | medium vegetation  | 
|5  | high vegetation  | 
|6  | building  | 
|7  | low point  | 
|8  | reserved  | 
|9  | water  | 
|10  | rail  | 
|11  | road surface  | 
|12  | reserved  | 
|13  | wire-guard(shield)  | 
|14  | wire-conductor(phase)  | 
|15  | transmission tower  | 
|16  | wire-structure connector(insulator)  | 
|17  | bridge deck  | 
|18  | high noise  | 
|19-63  |reserved   | 
|64-255  |user definable   | 


Many python libraries deal with point cloud data, commonly used, including [PDAL](https://pdal.io/)，[PCL](https://pointclouds.org/)，[open3D](http://www.open3d.org/docs/release/introduction.html) etc. Among them, PDAL can handle the data in .las format. The data can be read and stored in other formats and can be processed using other libraries' capabilities.

The experimental data of Illinois[.las format of the lidar data](https://www.arcgis.com/apps/webappviewer/index.html?id=44eb65c92c944f3e8b231eb1e2814f4d) is published by the Illinois state geological survey - prairie research institute. The research target area is the City of Chicago and its surrounding areas; due to the 1m resolution, some data volume in the research area is as high as 1.4t. Each tile is  $2501 \times 2501$, about 1G. The smallest one is hundreds of M. For ordinary computer configurations, processing big data usually involves determining how much memory can support it(many programs can process data without reading it all into memory, such as batch writing and batch reading of data in h5py format; The Rasterio library provides windows functionality to partition to read individual raster files). As well as CPU speed, batch processing can avoid the loss of all data due to processing interruption. In the elaboration of point cloud data processing, it does not process all regional data of the City of Chicago. It only takes the data processing within a certain range of the IIT(Illinois Institute of Technology) campus as the core. The unit Number(files) included in the downloaded point cloud data are(a total of 73 cell files, accounting for 26.2GB):

|   |   |   |   |   ||||||
|---|---|---|---|---||||||
|LAS_16758900.las|LAS_17008900.las|LAS_17258900.las|LAS_17508900.las|LAS_17758900.las|LAS_18008900.las|LAS_18258900.las||||
|LAS_16758875.las|LAS_17008875.las|LAS_17258875.las|LAS_17508875.las|LAS_17758875.las|LAS_18008875.las|LAS_18258875.las||||
|LAS_16758850.las|LAS_17008850.las|LAS_17258850.las|LAS_17508850.las|LAS_17758850.las|LAS_18008850.las|LAS_18258850.las||||
|LAS_16758825.las|LAS_17008825.las|LAS_17258825.las|LAS_17508825.las|LAS_17758825.las|LAS_18008825.las|LAS_18258825.las||||
|LAS_16758800.las|LAS_17008800.las|LAS_17258800.las|LAS_17508800.las|LAS_17758800.las|LAS_18008800.las|LAS_18258800.las|LAS_18508800.las|||
|LAS_16758775.las|LAS_17008775.las|LAS_17258775.las|LAS_17508775.las|LAS_17758775.las|LAS_18008775.las|LAS_18258775.las|LAS_18508775.las|||
|LAS_16758750.las|LAS_17008750.las|LAS_17258750.las|LAS_17508750.las|LAS_17758750.las|LAS_18008750.las|LAS_18258750.las|LAS_18508750.las|LAS_18758750.las|
|LAS_16758725.las|LAS_17008725.las|LAS_17258725.las|LAS_17508725.las|LAS_17758725.las|LAS_18008725.las|LAS_18258725.las|LAS_18508725.las|LAS_18758725.las|LAS_19008725.las|
|LAS_16758700.las|LAS_17008700.las|LAS_17258700.las|LAS_17508700.las|LAS_17758700.las|LAS_18008700.las|LAS_18258700.las|LAS_18508700.las|LAS_18758700.las|LAS_19008700.las|

### 1.1 Point cloud data processing(.las)
#### 1.1.1 Views point cloud data information

* Main parameter configuration of PDAL(see the official website of PDAL or 'PDAL: Point cloud Data Abstraction Library')

1. [Dimensions](https://pdal.io/dimensions.html)，Dimension, this parameter gives different information that might be stored, "type" can be configured based on a dimension; for example, the dimension can be configured as "dimension": "X", then "type": "filters.sort", that is, the point cloud returned by sorting according to the given dimension. Common ones include 'Classification'(data classification), 'Density'(point density),  'GpsTime'(to obtain the GPS time of this point), 'Intensity'(the intensity of the object's reflective surface), X,Y,Z(coordinates). The following code pipeline.arrays return a list array containing: ` dtype=[('X', '<f8'), ('Y', '<f8'), ('Z', '<f8'), ('Intensity', '<u2'), ('ReturnNumber', 'u1'), ('NumberOfReturns', 'u1'), ('ScanDirectionFlag', 'u1'), ('EdgeOfFlightLine', 'u1'), ('Classification', 'u1'), ('ScanAngleRank', '<f4'), ('UserData', 'u1'), ('PointSourceId', '<u2'), ('GpsTime', '<f8'), ('ScanChannel', 'u1'), ('ClassFlags', 'u1')])]`,It can be clear which dimensions are included in .las point cloud.

2. [Filters](https://pdal.io/stages/filters.html)，can delete, modify, or reorganize data streams by given the way data is manipulated. Some filters need to be implemented on the corresponding dimensions, such as reprojection on XYZ coordinates. Commonly used filters are:   create part:   filters.approximatecoplanar, based on k nearest neighbor, the planarity is estimated; filters.cluster, euclidean distance measurement is used to extract and label clustering; filters.dbscan, density-based spatial clustering; filters.covariancefeatures, local features are calculated based on the covariance of a point neighborhood; filters.eigenvalues, the point's eigenvalue is calculated based on k-nearest neighbor; filters.nndistance, calculate the distance exponent according to the nearest neighbor; filters.radialdensity, the density of the points at a given distance. Order part: filters.mortonorder, using Morton to sort XY data; filters.randomize, randomized points in a view; filters.sort, sort data based on a given dimension. Move part: filters.reprojection, use GDAL to re-project data from one coordinate system to another; filters.transformation, use the 4x4 transformation matrix to transform each point. Cull part: filters.crop, according to the bounding box or a polygon, filter points; filters.iqr, to eliminate points outsides the quartile range on a given dimension; filters.locate, given a dimension, a point is returned by min/max; filters.sample, perform poisson sampling and return only a subset of the input points; filters.voxelcenternearestneighbor, returns the point within each voxel closest to the center of the voxel; filters.voxelcentroidnearestneighbor, returns the point within each voxel closest to the center of the voxel mass. Join part: filters.merge, merge data from two different readers into a single stream. Mesh part: creat mesh using Delaunay  triangulation; filters.gridprojection, mesh projection method is used to create mesh; filters.poisson, create an object using the Poisson surface reconstruction algorithm. Languages part: filters.python, embed python code in the pipeline. Metadata part: filters.stats, calculate statistics for each dimension(mean, minimum, maximum, and so on).

3. type-[readers](https://pdal.io/stages/readers.html)-[writers](https://pdal.io/stages/writers.html)，For example, by `"type":"writers.gdal"`, using `"gdaldriver":"GTiff` drive, the raster data can be created from the point cloud based on the difference algorithm. Common types of data saved are: writers.gdal，writers.las，writers.ogr，writers.pgpointcloud，writers.ply，writers.sqlite，writers.text, and so on.

4. type, usually used with Filters. For example, if you configure "type":"filters.crop", you can set "bounds":"([0,100],[0,100])" boundary box to crop.

5. output_type, is the manner in which data calculation is given, such as mean,min,max,idx,count,stdev,all,idw, etc.

6.  resolution, specifies the accuracy of the output raster, such as 1,10, etc.

7. filename，specified a name to save the file.

8. [data_type](https://pdal.io/types.html)，type of data saved, for example int8,int16,unint8,float,double, etc. 

9. limits, data restrictions, such as configuring filters to "type":"filters.range", 则"limits":"Z[0:],Classification[6:6]"，only extract points with mark 6, namely the point of building classification, and the z value of the building is extracted


PDAL is a command-line utility. Open the terminal on Anaconda and enter the following command to get the point information. The command-line operation mode can avoid large quantities of data read into memory, causing an overflow. However, it is not convenient to view the data, so which way to use can be determined according to the specific situation.

pdal info F:\GitHubBigData\IIT_lidarPtClouds\rawPtClouds\LAS_16758900.las -p 0{
  "file_size": 549264685,
  "filename": "F:\\GitHubBigData\\IIT_lidarPtClouds\\rawPtClouds\\LAS_16758900.las",
  "now": "2020-08-15T18:56:44-0500",
  "pdal_version": "2.1.0 (git-version: Release)",
  "points":
  {
    "point":
    {
      "ClassFlags": 0,
      "Classification": 3,
      "EdgeOfFlightLine": 0,
      "GpsTime": 179803760,
      "Intensity": 1395,
      "NumberOfReturns": 1,
      "PointId": 0,
      "PointSourceId": 0,
      "ReturnNumber": 1,
      "ScanAngleRank": 15,
      "ScanChannel": 0,
      "ScanDirectionFlag": 0,
      "UserData": 0,
      "X": 1167506.44,
      "Y": 1892449.7,
      "Z": 594.62
    }
  },
  "reader": "readers.las"
}
* pipeline

In point cloud data processing, it usually includes reading, processing, writing, and other operations. To facilitate the process, PDAL introduces a pipeline concept that stacks multiple operations into a data flow defined by a JSON data format; this is especially advantageous for complex processing processes. PDAL also provides the python schema, which enables the PDAL library to be called into python, and defines the pipeline operation flow. For example, a simple case provided by the official website includes reading the .las file("%s"%separate_las), configuring the dimension as the point cloud coordinate("dimension": "X"), and sorting the returned array according to the x coordinate("type": "filters.sort"), etc. After executing pipeline（`pipeline.execute()`, the pipeline object returns the point cloud with the dimension value. Its 'dtypes' item returns the point cloud's dimension, corresponding to the returned array information. The number of points in this cell is count=18721702 points.

The metadata can be self-printed for viewing, including coordinate projection information.


```python
import util
dirpath=r"F:\GitHubBigData\IIT_lidarPtClouds\rawPtClouds"
fileType=["las"]
las_paths=util.filePath_extraction(dirpath,fileType)

s_t=util.start_time()
import pdal,os
separate_las=os.path.join(list(las_paths.keys())[0],list(las_paths.values())[0][32]).replace("\\","/") #Note that "\" and "/" in the filename path may vary in the types supported by different libraries and need to be adjusted.

json="""
[
    "%s",
    {
        "type": "filters.sort",
        "dimension": "X"
    }
]
"""%separate_las

pipeline=pdal.Pipeline(json)
count=pipeline.execute()
print("pts count:",count)
arrays=pipeline.arrays
print("arrays:",arrays)
metadata=pipeline.metadata
log=pipeline.log
print("complete .las reading ")
util.duration(s_t)
```

    start time: 2020-08-15 23:08:37.251949
    pts count: 16677942
    arrays: [array([(1175000., 1884958.18, 634.81,  9832, 1, 1, 0, 0, 5, 15., 0, 0, 1.78265216e+08, 0, 0),
           (1175000., 1884941.11, 644.75, 18604, 1, 1, 0, 0, 5, 15., 0, 0, 1.78265216e+08, 0, 0),
           (1175000., 1884931.3 , 641.59,  4836, 1, 1, 0, 0, 5, 15., 0, 0, 1.78265232e+08, 0, 0),
           ...,
           (1177500., 1882882.43, 597.43, 50337, 1, 1, 0, 0, 2, 15., 0, 0, 1.78265200e+08, 0, 0),
           (1177500., 1882865.16, 597.49, 44165, 1, 1, 0, 0, 2, 15., 0, 0, 1.78239520e+08, 0, 0),
           (1177500., 1882501.12, 596.34, 44193, 1, 1, 0, 0, 2, 15., 0, 0, 1.78239520e+08, 0, 0)],
          dtype=[('X', '<f8'), ('Y', '<f8'), ('Z', '<f8'), ('Intensity', '<u2'), ('ReturnNumber', 'u1'), ('NumberOfReturns', 'u1'), ('ScanDirectionFlag', 'u1'), ('EdgeOfFlightLine', 'u1'), ('Classification', 'u1'), ('ScanAngleRank', '<f4'), ('UserData', 'u1'), ('PointSourceId', '<u2'), ('GpsTime', '<f8'), ('ScanChannel', 'u1'), ('ClassFlags', 'u1')])]
    complete .las reading 
    end time: 2020-08-15 23:09:09.810355
    Total time spend:0.53 minutes
    

After PDAL processing, the properties of the 'pipeline' object can be read. Because a point contains more than one piece of information, we can convert the point cloud array to DataFrame format data for easy viewing.


```python
import pandas as pd
pts_df=pd.DataFrame(arrays[0])
util.print_html(pts_df)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X</th>
      <th>Y</th>
      <th>Z</th>
      <th>Intensity</th>
      <th>ReturnNumber</th>
      <th>NumberOfReturns</th>
      <th>ScanDirectionFlag</th>
      <th>EdgeOfFlightLine</th>
      <th>Classification</th>
      <th>ScanAngleRank</th>
      <th>UserData</th>
      <th>PointSourceId</th>
      <th>GpsTime</th>
      <th>ScanChannel</th>
      <th>ClassFlags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1175000.0</td>
      <td>1884958.18</td>
      <td>634.81</td>
      <td>9832</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>178265216.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1175000.0</td>
      <td>1884941.11</td>
      <td>644.75</td>
      <td>18604</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>178265216.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1175000.0</td>
      <td>1884931.30</td>
      <td>641.59</td>
      <td>4836</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>178265232.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1175000.0</td>
      <td>1884936.81</td>
      <td>644.47</td>
      <td>15047</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>179806048.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1175000.0</td>
      <td>1884948.02</td>
      <td>635.57</td>
      <td>2700</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>15.0</td>
      <td>0</td>
      <td>0</td>
      <td>179805664.0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>



In addition to directly viewing the data content, The open3D library can print 3d point cloud for interactive observation. However, since PDAL was originally used to read .las format point cloud(Open3D does not currently support reading .las format point cloud data), the read point cloud data needs to be converted to the open3D supported format. The displayed colors represent the height information of the point cloud.


```python
import open3d as o3d
o3d_pts=o3d.geometry.PointCloud()
o3d_pts.points=o3d.utility.Vector3dVector(pts_df[['X','Y','Z']].to_numpy())
o3d.visualization.draw_geometries([o3d_pts])
```

<<a href=""><img src="./imgs/12_01.png" height="auto" width="auto" title="caDesign"></a>

#### 1.1.2 Establish DSM(Digital Surface Model) and classification raster
The 3D point cloud is 3d format data, which can be extracted and converted into corresponding 2D raster data for convenient data analysis. The analysis 3D point cloud data will be further explored in relevant chapters.  The most commonly used transformation of point cloud data into 2d rater data includes the generation of classification raster data, namely, land cover type; Secondly, the height of ground objects is extracted, such as the height of buildings and the height of vegetation; Third, Digital Elevation Model(DEM) is generated, and so on.

DEM-The digital elevation model is the surface elevation of bare ground that removed natural and architectural objects；

DTM-The digital terrain model adds vector features of natural terrains, such as rivers and ridges, to DEM. DEM and DTM are not clearly distinguished in many cases, determined by the data's content.

DSM-The digital surface model captures both ground, natural(such as trees), and human-made (such as buildings) features.

The content to be processed by point cloud data is defined in a function. Each processing content is a pipeline, defined in JSON format. The following code defines three contents. One is to extract land cover classification information for establishing a classification raster; Secondly, extract elevation information to establish DSM; Third, only the elevation of ground surface type can be extracted to establish DEM. The input parameter json_combo is defined to manage and judge the pipeline to be calculated to facilitate the continuous addition in the function of new extraction content in the future, to flexibly handle the function and avoid big changes when a new pipeline is added in the future.

Geospatial data with large files are usually placed on the hard disk after a major data processing is completed during the processing process, and then read when used. So after processing a point cloud unit(tile), it is immediately saved to the hard disk instead of residing in memory to avoid memory overflow.



```python
import os

def las_info_extraction(las_fp,json_combo):
    import pdal
    '''
    function - Convert single .las point cloud data into classification raster data and DSM raster data, etc.
    
    Paras:
    las_fp - .las format file path
    save_path - Saved path list, classification, and DSM storage under different paths
    
    '''
    pipeline_list=[]
    if 'json_classification' in json_combo.keys():
        #pipeline-Used to create a classification raster
        json_classification="""
            {
            "pipeline": [
                         "%s",
                         {
                         "filename":"%s",
                         "type":"writers.gdal",
                         "dimension":"Classification",
                         "data_type":"uint16_t",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_classification'])
        pipeline_list.append(json_classification)
        
    elif 'json_DSM' in json_combo.keys():
        #pipeline-Used to build DSM raster data
        json_DSM="""
            {
            "pipeline": [
                         "%s",
                         {
                         "filename":"%s",
                         "gdaldriver":"GTiff",
                         "type":"writers.gdal",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_DSM']) 
        pipeline_list.append(json_DSM)
        
    elif 'json_ground' in json_combo.keys():
        #pipelin-Used to extract the ground surface
        json_ground="""
            {
            "pipeline": [
                         "%s",
                         {
                         "type":"filters.range",
                         "limits":"Classification[2:2]"                     
                         },
                         {
                         "filename":"%s",
                         "gdaldriver":"GTiff",
                         "type":"writers.gdal",
                         "output_type":"mean",  
                         "resolution": 1
                         }               
            ]        
            }"""%(las_fp,json_combo['json_ground'])   
        pipeline_list.append(json_ground)
    
    
    for json in pipeline_list:
        pipeline=pdal.Pipeline(json)
        pipeline.loglevel=8 #Log level configuration
        if pipeline.validate(): #Check that if the JSON option is correct.
            #print(pipeline.validate())
            try:
                count=pipeline.execute()
            except:
                print("\n An exception occurred,the file name:%s"%las_fp)
                print(pipeline.log) #If an error occurs, print the log to view it so we can fix the error code.
                
        else:
            print("pipeline unvalidate!!!")
    print("finished conversion...")
            
dirpath=r"F:\GitHubBigData\IIT_lidarPtClouds\rawPtClouds"            
las_fp=os.path.join(dirpath,'LAS_17508825.las').replace("\\","/")
workspace=r'F:\GitHubBigData\IIT_lidarPtClouds'
json_combo={"json_classification":os.path.join(workspace,'classification_DSM\LAS_17508825_classification.tif').replace("\\","/"),"json_DSM":os.path.join(workspace,'classification_DSM\LAS_17508825_DSM.tif').replace("\\","/")} #Configure input parameters
las_info_extractio(las_fp,json_combo)
```

    finished conversion...
    

The saved DSM raster files were read and printed with the Earthpy library for viewing. There might be outliers in the data, resulting in grayscale on display; vmin and vmax parameters could be configured by quantile(np.quantile) method.


```python
import rasterio as rio
import os
import numpy as np
import earthpy.plot as ep

workspace=r'F:\GitHubBigData\IIT_lidarPtClouds'
with rio.open(os.path.join(workspace,'classification_DSM\LAS_17508825_DSM.tif')) as DSM_src:
    DSM_array=DSM_src.read(1)
titles = ["LAS_17508825_DTM"]
ep.plot_bands(DSM_array, cmap="turbo", cols=1, title=titles, vmin=np.quantile(DSM_array,0.1), vmax=np.quantile(DSM_array,0.9))
```


<<a href=""><img src="./imgs/12_02.png" height="auto" width="auto" title="caDesign"></a>





    <AxesSubplot:title={'center':'LAS_17508825_DTM'}>



The saved category-raster data is also read, but we need to define our print-display function to print according to the categories indicated by the integers. The category is determined by the classification identifier given by LAS format, and the color can be defined by itself according to the effect to be achieved by the display. Add added legend, convenient to see the color of the corresponding classification.


```python
def las_classification_plotWithLegend(las_fp):  
    import rasterio as rio
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib import colors
    from matplotlib.patches import Rectangle
    '''
    function - Displays the classification raster file generated by the .las file and displays the legend.
    
    Paras:
    las_fp - Category file path
    '''    
    with rio.open(las_fp) as classi_src:
        classi_array=classi_src.read(1)

    las_classi_colorName={0:'black',1:'white',2:'beige',3:'palegreen',4:'lime',5:'green',6:'tomato',7:'silver',8:'grey',9:'lightskyblue',10:'purple',11:'slategray',12:'grey',13:'cadetblue',14:'lightsteelblue',15:'brown',16:'indianred',17:'darkkhaki',18:'azure',9999:'white'}
    las_classi_colorRGB=pd.DataFrame({key:colors.hex2color(colors.cnames[las_classi_colorName[key]]) for key in las_classi_colorName.keys()})
    classi_array_color=[pd.DataFrame(classi_array).replace(las_classi_colorRGB.iloc[idx]).to_numpy() for idx in las_classi_colorRGB.index]
    classi_array_color_=np.concatenate([np.expand_dims(i,axis=-1) for i in classi_array_color],axis=-1)
    fig, ax=plt.subplots(figsize=(12, 12))
    im=ax.imshow(classi_array_color_, )
    ax.set_title(
        "LAS_classification",
        fontsize=14,
    )

    #Add the legend
    color_legend=pd.DataFrame(las_classi_colorName.items(),columns=["id","color"])
    las_classi_name={0:'never classified',1:'unassigned',2:'ground',3:'low vegetation',4:'medium vegetation',5:'high vegetation',6:'building',7:'low point',8:'reserved',9:'water',10:'rail',11:'road surface',12:'reserved',13:'wire-guard(shield)',14:'wire-conductor(phase)',15:'transimission',16:'wire-structure connector(insulator)',17:'bridge deck',18:'high noise',9999:'null'}
    color_legend['label']=las_classi_name.values()
    classi_lengend=[Rectangle((0, 0), 1, 1, color=c) for c in color_legend['color']]

    ax.legend(classi_lengend,color_legend.label,mode='expand',ncol=3)
    plt.tight_layout()
    plt.show()

import os
workspace=r'F:\GitHubBigData\IIT_lidarPtClouds'
las_fp=os.path.join(workspace,'classification_DSM\LAS_17508825_classification.tif')
las_classification_plotWithLegend(las_fp)   
```


<<a href=""><img src="./imgs/12_03.png" height="auto" width="auto" title="caDesign"></a>


#### 1.1.3Batch processing .las point cloud unit(tile)
The point cloud data processing of a unit is completed through the code debugging of a point cloud unit. A batch processing function of point cloud data is established to batch processing all point cloud units; This function directly calls the above single unit processing function of point cloud and only combs the read and save path of all point unit files. The tqdm library allows us to display the loop calculation as a progress bar to see the calculation's processing,  specifying the approximate time it will take to complete all the data calculations. Meanwhile, start_time() and duration(s_t) methods defined in the OpenStreetMap section were called to calculate the specific duration.



```python
import util
def las_info_extraction_combo(las_dirPath,json_combo_):
    import util,os,re
    from tqdm import tqdm
    '''
    function - Batch conversion .las point cloud data into DSM and classification raster.
    
    Paras:
    las_dirPath - LAS file path
    save_path - save path
    
    return:
        
    '''
    file_type=['las']
    las_fn=util.filePath_extraction(las_dirPath,file_type)
    '''Flattening list function'''
    flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]
    las_fn_list=flatten_lst([[os.path.join(k,las_fn[k][i]) for i in range(len(las_fn[k]))] for k in las_fn.keys()])
    pattern=re.compile(r'[_](.*?)[.]', re.S) 
    for i in tqdm(las_fn_list):  
        fn_num=re.findall(pattern, i.split("\\")[-1])[0] #Extract the number in the filename string
        #Note that "\" and "/" in the file name path may vary in the types by different libraries and need to be adjusted.
        json_combo={key:os.path.join(json_combo_[key],"%s_%s.tif"%(os.path.split(json_combo_[key])[-1],fn_num)).replace("\\","/") for key in json_combo_.keys()}        
        util.las_info_extraction(i.replace("\\","/"),json_combo)
    
dirpath=r"F:\GitHubBigData\IIT_lidarPtClouds\rawPtClouds"    
json_combo_={"json_classification":r'F:\GitHubBigData\IIT_lidarPtClouds\classification'.replace("\\","/"),"json_DSM":r'F:\GitHubBigData\IIT_lidarPtClouds\DSM'.replace("\\","/")} #Configure input parameters

s_t=util.start_time()S
las_info_extraction_combo(dirpath,json_combo_)  
util.duration(s_t)
```

      0%|          | 0/73 [00:00<?, ?it/s]

    start time: 2020-08-16 14:04:13.539920
    

    100%|██████████| 73/73 [18:15<00:00, 15.01s/it]

    end time: 2020-08-16 14:22:28.926273
    Total time spend:18.25 minutes
    

    
    

* Merge raster data

After batch processing of all point cloud data in point cloud units, that is, generating multiple DSM files and multiple classification files with the same number of point cloud units, it is necessary to merge them into a complete raster file. The merge method provided by the Rasterio library is mainly used. It is also important to configure the compression and the saving type; otherwise the merged raster file may be very large. For example, the size of the file is about 4.5GB after merging all the raters. However, with `"compress":'lzw' and `"dtype":get_minimum_int_dtype(mosaic), `, the file size is only 201MB, which reduces the size of the file to save disk space and memory speed.

When the configuration file is saved, the function `get_minimum_int_dtype(values)` given by the Rasterio library is transferred to determine the file type to be saved based on the array value, thus avoiding self-definition.


```python
def raster_mosaic(dir_path,out_fp,):
    import rasterio,glob,os
    from rasterio.merge import merge
    '''
    function - Merge multiple rasters into one
    
    Paras:
    dir_path - Raster root directory
    out-fp - Saving path
    
    return:
    out_trans - Return transfermation informaiton
    '''
    
    #Transfer the function provided by Rasterio that defines the data type with the minimum array size. 
    def get_minimum_int_dtype(values):
        """
        Uses range checking to determine the minimum integer data type required
        to represent values.

        :param values: numpy array
        :return: named data type that can be later used to create a numpy dtype
        """

        min_value = values.min()
        max_value = values.max()

        if min_value >= 0:
            if max_value <= 255:
                return rasterio.uint8
            elif max_value <= 65535:
                return rasterio.uint16
            elif max_value <= 4294967295:
                return rasterio.uint32
        elif min_value >= -32768 and max_value <= 32767:
            return rasterio.int16
        elif min_value >= -2147483648 and max_value <= 2147483647:
            return rasterio.int32
    
    search_criteria = "*.tif" #Search for the .tif raster to be merged
    fp_pattern=os.path.join(dir_path, search_criteria)
    fps=glob.glob(fp_pattern) #Use the glob library to search for files with the specified pattern.
    src_files_to_mosaic=[]
    for fp in fps:
        src=rasterio.open(fp)
        src_files_to_mosaic.append(src)    
    mosaic,out_trans=merge(src_files_to_mosaic)  #The merge function returns an array of rasters and conversion information.
    
    #Get metadata
    out_meta=src.meta.copy()
    #Update metadata
    data_type=get_minimum_int_dtype(mosaic)
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     #Reduce the storage file size by compressing and configuring the storage type
                     "compress":'lzw',
                     "dtype":get_minimum_int_dtype(mosaic), 
                      }
                    )       
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic.astype(data_type))     
    
    return out_trans
DSM_dir_path=r'F:\GitHubBigData\IIT_lidarPtClouds\DSM'
DSM_out_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\DSM_mosaic.tif'

import util
s_t=util.start_time()
out_trans=raster_mosaic(DSM_dir_path,DSM_out_fp)
util.duration(s_t)
```

    start time: 2020-08-17 10:06:56.893560
    end time: 2020-08-17 10:07:53.994923
    Total time spend:0.95 minutes
    

Read, print, and view the merged DSM raster in the same manner as described above.


```python
import rasterio as rio
import earthpy.plot as ep
import numpy as np
DSM_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\DSM_mosaic.tif'
with rio.open(DSM_fp) as DSM_src:
    mosaic_DSM_array=DSM_src.read(1)
titles = ["mosaic_DTM"]
ep.plot_bands(mosaic_DSM_array, cmap="turbo", cols=1, title=titles, vmin=np.quantile(mosaic_DSM_array,0.25), vmax=np.quantile(mosaic_DSM_array,0.95))
```


<<a href=""><img src="./imgs/12_04.png" height="auto" width="auto" title="caDesign"></a>





    <AxesSubplot:title={'center':'mosaic_DTM'}>



Also merges a single classification raster into a single file


```python
classi_dir_path=r'F:\GitHubBigData\IIT_lidarPtClouds\classification'
classi_out_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\classification_mosaic.tif'

import util
s_t=util.start_time()
out_trans=util.raster_mosaic(classi_dir_path,classi_out_fp)
util.duration(s_t)
```

    start time: 2020-08-16 16:26:06.796357
    end time: 2020-08-16 16:26:29.746152
    Total time spend:0.37 minutes
    


```python
import rasterio as rio
import earthpy.plot as ep
import numpy as np
classi_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\classification_mosaic.tif'
with rio.open(classi_fp) as classi_src:
    mosaic_classi_array=classi_src.read(1)

from skimage.transform import rescale
mosaic_classi_array_rescaled=rescale(mosaic_classi_array, 0.2, anti_aliasing=False,preserve_range=True)
print("original shape:",mosaic_classi_array.shape)
print("rescaled shape:",mosaic_classi_array_rescaled.shape)

import util
util.las_classification_plotWithLegend_(mosaic_classi_array_rescaled)
```

    original shape: (22501, 25001)
    rescaled shape: (4500, 5000)
    

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


<<a href=""><img src="./imgs/12_05.png" height="auto" width="auto" title="caDesign"></a>


### 1.2 Building height extraction
The process of building height extraction is as follows: DSM raster is re-projected into the coordinate projection system defined by Landsat in this area(Chicago), and the projection coordinate system is unified; -> The .shp format(Polygon) buildings distribution data 'Building Footprints (current)' is obtained from  [Chicago Data Portal，CDP](https://data.cityofchicago.org/); -> .shp format building distribution Polygon is cropped according to the DSM raster extent, and projected as the same as the DSM raster re-projected; -> Extracts the ground information using PSAL, and save as raster; -> Interpolates all single ground raster, merge them, re-projected them as the same as the DSM projection, and then save them after projection; ->The elevation data of the building area is extracted from DSM according to the classified raster data; ->The 'zonal_stats' method provided by the 'rasterstats' library was used to extract elevation information of DSM and ground raster respectively, with the statistical methods such as median; -> DSM-ground is extracted using zone statistics, namely, the building height data; ->  Write the building height data into GeoDataFrame and save it as .shp file for later analysis.

#### 1.2.1 Defines the get raster projection function and the raster reprojection function

Projection and reprojection methods have been used in Landsat remote sensing image processing, which can be viewed in combination.


```python
def get_crs_raster(raster_fp):
    import rasterio as rio
    '''
    function - Gets the projected coordinates of the given raster-crs.
    
    Paras:
    raster)fp - The path for a given raster file
    '''
    with rio.open(raster_fp) as raster_crs:
        raster_profile=raster_crs.profile
        return raster_profile['crs']
    
ref_raster=r'F:\data_02_Chicago\9_landsat\data_processing\DE_Chicago.tif'  # For remote sensing images processed in the Landsat section, Landsat of the corresponding region can be downloaded and used as an input parameter to obtain its projection. 
dst_crs=get_crs_raster(ref_raster)
print("dst_crs:",dst_crs)
```

    dst_crs: EPSG:32616
    


```python
def raster_reprojection(raster_fp,dst_crs,save_path):
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    import rasterio as rio
    '''
    function - Transform raster projection
    
    Paras:
    raster_fp - The raster to be transformed
    dst_crs - Target projection
    save_path - Saving path
    '''
    with rio.open(raster_fp) as src:
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rio.open(save_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)      
    print("finished reprojecting...")
   
DTM_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\DTM_mosaic.tif'
DTM_reprojection_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\DTM_mosaic_reprojection.tif'
dst_crs=dst_crs
raster_reprojection(DTM_fp,dst_crs,DTM_reprojection_fp)  
```

    finished reprojecting...
    

#### 1.2.2 Get the extent of the raster given to crop .shp format file(Polygon)

In the case of vector data cropping with 'gpd.clip(vector_projection_,poly_gdf) ' method, data need to ben cleaned up, including 'gpd.clip(vector_projection_,poly_gdf) ' to cleanse the null values; And `polygon_bool=vector_projection.geometry.apply(lambda row:True if type(row)==type_Polygon and row.is_valid else False)` to cleanse invalid Polygon object, and not for 'shapely.geometry.polygon.Polygon' format.  Only after the data has been cleaned up can the crop be performed, otherwise an error will occur.


```python
def clip_shp_withRasterExtent(vector_shp_fp,reference_raster_fp,save_path):
    import rasterio as rio
    from rasterio.plot import plotting_extent
    import geopandas as gpd    
    import pandas as pd
    from shapely.geometry import Polygon
    import shapely
    '''
    function - Crop data in .shp format to the extent of a given raster and define a projection as the same as the raster.
    
    Paras:
    vector_shp_fp - Vector file path to be cropped
    reference_raster_fp - Reference raster, extent and projection
    save_path - Save the path
    
    return:
    poly_gdf - Returns the cropped boundary
    '''
    vector=gpd.read_file(vector_shp_fp)
    print("before dropna:",vector.shape)
    vector.dropna(subset=["geometry"], inplace=True)
    print("after dropna:",vector.shape)
    with rio.open(reference_raster_fp) as src:
        raster_extent=plotting_extent(src)
        print("extent:",raster_extent)
        raster_profile=src.profile
        crs=raster_profile['crs']
        print("crs:",crs)        
        polygon=Polygon([(extent[0],extent[2]),(extent[0],extent[3]),(extent[1],extent[3]),(extent[1],extent[2]),(extent[0],extent[2])])
        #poly_gdf=gpd.GeoDataFrame([1],geometry=[polygon],crs=crs)  
        poly_gdf=gpd.GeoDataFrame({'name':[1],'geometry':[polygon]},crs=crs)  
        vector_projection=vector.to_crs(crs)
     
    #Remove row with a non-Polygon type and invalid Polygon(validated with .is_valid), or the clip cannot be performed.
    type_Polygon=shapely.geometry.polygon.Polygon
    polygon_bool=vector_projection.geometry.apply(lambda row:True if type(row)==type_Polygon and row.is_valid else False)
    vector_projection_=vector_projection[polygon_bool]

    vector_clip=gpd.clip(vector_projection_,poly_gdf)    
    vector_clip.to_file(save_path)
    print("finished clipping and projection...")
    return poly_gdf
    
DTM_reprojection_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\DTM_mosaic_reprojection.tif'  
vector_shp_fp=r'F:\GitHubBigData\geo_data\Building Footprints.shp'
save_path=r'F:\GitHubBigData\geo_data\building_footprints_clip_projection.shp'
poly_gdf=clip_shp_withRasterExtent(vector_shp_fp,DTM_reprojection_fp,save_path)
```

    before dropna: (820606, 50)
    after dropna: (820600, 50)
    extent: (445062.87208577903, 452785.6657144044, 4627534.041486926, 4634507.01087126)
    crs: EPSG:32616
    finished clipping and projection...
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>POLYGON ((445062.872 4627534.041, 445062.872 4...</td>
    </tr>
  </tbody>
</table>
</div>



* View the processed .shp format building vector data

It is possible to overlay print DSM raster data and building vector data, which determine that they are consistent under the consistent geospatial coordinates system to show that the data processing is correct; otherwise, we need to go back to see the previous code, determine the location of the error, adjust the code to recalculate.


```python
import matplotlib.pyplot as plt
import geopandas as gpd
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
vector=gpd.read_file(vector_shp_fp)
vector.plot(ax=ax1)
```


```python
import matplotlib.pyplot as plt
import earthpy.plot as ep
import rasterio as rio
from rasterio.plot import plotting_extent
import numpy as np
import geopandas as gpd

fig, ax=plt.subplots(figsize=(12, 12))

DTM_reprojection_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\DTM_mosaic_reprojection.tif'
with rio.open(DTM_reprojection_fp) as DTM_src:
    mosaic_DTM_array=DTM_src.read(1)
    plot_extent=plotting_extent(DTM_src)
    
titles = ["building and DTM"]
ep.plot_bands(mosaic_DTM_array, cmap="binary", cols=1, title=titles, vmin=np.quantile(mosaic_DTM_array,0.25), vmax=np.quantile(mosaic_DTM_array,0.95),ax=ax,extent=plot_extent)

building_clipped_fp=r'F:\GitHubBigData\geo_data\building_footprints_clip_projection.shp'
vector=gpd.read_file(building_clipped_fp)
vector.plot(ax=ax,color='tomato')
plt.show()
```


<<a href=""><img src="./imgs/12_06.png" height="auto" width="auto" title="caDesign"></a>


#### 1.2.3 The elevation data of the building area is extracted from DSM according to the classified raster data.

Because each building polygon in the building vector data does not necessarily include only the DSM raster classified as a building, it may include other classified data. Therefore, DSM is required only to retain building elevation information to avoid calculation errors. It is implemented with np.where(). 


```python
import util
classi_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\classification_mosaic.tif'
classi_reprojection_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\classi_mosaic_reprojection.tif'
dst_crs=dst_crs
util.raster_reprojection(classi_fp,dst_crs,classi_reprojection_fp)  
```

    finished reprojecting...
    


```python
import util
s_t=util.start_time()  

classi_reprojection_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\classi_mosaic_reprojection.tif'
with rio.open(classi_reprojection_fp) as classi_src:
    classi_reprojection=classi_src.read(1)
    out_meta=classi_src.meta.copy()
    
building_DSM=np.where(classi_reprojection==6,mosaic_DTM_array,np.nan) #Only building elevation information is retained.
building_DSM_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\building_DSM.tif'
with rio.open(building_DSM_fp, "w", **out_meta) as dest:
    dest.write(building_DSM.astype(rio.uint16),1)     
util.duration(s_t)
```

    start time: 2020-08-17 01:36:21.210867
    end time: 2020-08-17 01:37:06.385012
    Total time spend:0.75 minutes
    

* Extract the ground and interpolate, merge, re-project, and see the data. 

-Extract


```python
import util
las_dirPath=r"F:\GitHubBigData\IIT_lidarPtClouds\rawPtClouds"
json_combo_={"json_ground":r'F:\GitHubBigData\IIT_lidarPtClouds\ground'}
util.las_info_extraction_combo(las_dirPath,json_combo_)
```

    100%|██████████| 73/73 [07:44<00:00,  6.37s/it]
    

-Interpolate

The interpolation uses the 'fillnodata' method provided by the Rasterio library; this method is to search the values conically in four directions for each pixel and calculate interpolation based on the weighted inverse distance. Once all the interpolation is completed, the 3x3 averaging filter can be used to iterate over the interpolating pixels, smoothing the data; This algorithm is usually suitable for continuously changing raster, such as DEM, and for filling small holes.


```python
def rasters_interpolation(raster_path,save_path,max_search_distance=400,smoothing_iteration=0):
    import rasterio,os
    from rasterio.fill import fillnodata
    import glob
    from tqdm import tqdm
    '''  
    function - The interpolation method 'rasterio.fill' is used to complete the missing data. 

    Paras:
    raster_path - Raster root directory
    save_path - Saved directory
    '''
    search_criteria = "*.tif" #Search for the .tif raster file to be merged
    fp_pattern=os.path.join(raster_path, search_criteria)
    fps=glob.glob(fp_pattern) #Use the glob library to search for files with the specified pattern.
    
    for fp in tqdm(fps):
        with rasterio.open(fp,'r') as src:
            data=src.read(1, masked=True)
            msk=src.read_masks(1) 
            #Configure the 'max_search_distance' parameter or perform interpolation multiple times to complete large missing data areas.
            fill_raster=fillnodata(data,msk,max_search_distance=max_search_distance,smoothing_iterations=0) 
            out_meta=src.meta.copy()   
            with rasterio.open(os.path.join(save_path,"interplate_%s"%os.path.basename(fp)), "w", **out_meta) as dest:            
                dest.write(fill_raster,1)
    
raster_path=r'F:\GitHubBigData\IIT_lidarPtClouds\ground'
save_path=r'F:\GitHubBigData\IIT_lidarPtClouds\ground_interpolation' 
import util
s_t=util.start_time()                            
rasters_interpolation(raster_path,save_path,max_search_distance=400,smoothing_iteration=0)    
util.duration(s_t)   
```

    start time: 2020-08-17 10:46:53.418270
    

    100%|██████████| 73/73 [05:33<00:00,  4.57s/it]

    end time: 2020-08-17 10:52:27.624715
    Total time spend:5.57 minutes
    

    
    

-Merge


```python
ground_dir_path=r'F:\GitHubBigData\IIT_lidarPtClouds\ground_interpolation' 
ground_out_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\ground_mosaic.tif'

import util
s_t=util.start_time()
out_trans=util.raster_mosaic(ground_dir_path,ground_out_fp)
util.duration(s_t)
```

    start time: 2020-08-17 10:52:38.334055
    end time: 2020-08-17 10:52:58.972188
    Total time spend:0.33 minutes
    

-Re-project


```python
import util
ref_raster=r'F:\data_02_Chicago\9_landsat\data_processing\DE_Chicago.tif'  # The remote sensing images processed in the Landsat section, the Landsat data corresponding region, can be downloaded and used as the input parameter to obtain its projection.
dst_crs=util.get_crs_raster(ref_raster)
print("dst_crs:",dst_crs)

ground_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\ground_mosaic.tif'
ground_reprojection_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\ground_mosaic_reprojection.tif'
util.raster_reprojection(ground_fp,dst_crs,ground_reprojection_fp)  
```

    dst_crs: EPSG:32616
    finished reprojecting...
    

-See the data

It is defined as a function to be called to facilitate the printing of raster data. 


```python
def raster_show(raster_fp,title='raster',vmin_vmax=[0.25,0.95],cmap="turbo"):
    import rasterio as rio
    import earthpy.plot as ep
    import numpy as np
    '''
    function - Display remote sensing images(one band) using the Earthpy library.
    
    Paras:
    raster_fp - The raster path input
    vmin_vmax -Adjust display interval
    '''   
    
    with rio.open(raster_fp) as src:
        array=src.read(1)
    titles=[title]
    ep.plot_bands(array, cmap=cmap, cols=1, title=titles, vmin=np.quantile(array,vmin_vmax[0]), vmax=np.quantile(array,vmin_vmax[1]))

raster_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\ground_mosaic_reprojection.tif'
raster_show(raster_fp)
```


<<a href=""><img src="./imgs/12_07.png" height="auto" width="auto" title="caDesign"></a>


#### 1.2.4 Zonal statistics, calculating the building height

DSM and ground elevation data are extracted using the 'zonal_stats' method of the Rasterstats library.


```python
from rasterstats import zonal_stats
import util
building_clipped_fp=r'F:\GitHubBigData\geo_data\building_footprints_clip_projection.shp'

s_t=util.start_time()  
building_DTM_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\building_DTM.tif'
stats_DTM=zonal_stats(building_clipped_fp, building_DTM_fp,stats="median")
ground_mosaic_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\ground_mosaic_reprojection.tif'
stats_ground=zonal_stats(building_clipped_fp, ground_mosaic_fp,stats="median")
util.duration(s_t)    
```

    start time: 2020-08-17 11:46:28.454714
    end time: 2020-08-17 11:51:01.105563
    Total time spend:4.53 minutes
    

Building height = elevation extracted by DSM - elevation extracted by ground. To facilitate the computation, convert it to the DataFrame data format provided by pandas, and apply '.apply' and lambda function to perform the computation. Moreover, add the calculation results to the GeoDataFrame of the building vector data, save as .shp format data.


```python
import numpy as np
import pandas as pd
import geopandas as gpd
building_height_df=pd.DataFrame({'dtm':[k['median'] for k in stats_DTM],'ground':[k['median'] for k in stats_ground]})
building_height_df['height']=building_height_df.apply(lambda row:row.dtm-row.ground if row.dtm>row.ground else -9999,axis=1)
print(building_height_df[:10])

building_clipped_fp=r'F:\GitHubBigData\geo_data\building_footprints_clip_projection.shp'
vector=gpd.read_file(building_clipped_fp)
vector['height']=building_height_df['height']
vector.to_file(r'F:\GitHubBigData\geo_data\building_footprints_height.shp')
print("finished computation and save...")
```

         dtm  ground  height
    0  617.0   595.0    22.0
    1  603.0   594.0     9.0
    2  639.0   601.0    38.0
    3  607.0   595.0    12.0
    4  600.0   590.0    10.0
    5  629.0   604.0    25.0
    6  599.0   589.0    10.0
    7  618.0   588.0    30.0
    8    0.0   594.0 -9999.0
    9  599.0   591.0     8.0
    finished computation and save...
    

Open and print to view the results.


```python
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

building_footprints_height_fp=r'F:\GitHubBigData\geo_data\building_footprints_height.shp'
building_footprints_height=gpd.read_file(building_footprints_height_fp)

fig, ax=plt.subplots(figsize=(12, 12))
divider=make_axes_locatable(ax)
cax_1=divider.append_axes("right", size="5%", pad=0.1)
building_footprints_height.plot(column='height',ax=ax,cax=cax_1,legend=True,cmap='OrRd',vmin=np.quantile(building_footprints_height.height,0.25), vmax=np.quantile(building_footprints_height.height,0.95)) #'OrRd','PuOr'
```




    <AxesSubplot:>




<<a href=""><img src="./imgs/12_08.png" height="auto" width="auto" title="caDesign"></a>


### 1.3 key point
#### 1.3.1 data processing technique

* Use libraries such as PDAL、open3d and PCL to process point cloud data.

#### 1.3.2 The newly created function tool

* function - Convert single .las point cloud data into classification raster data and DSM raster data, etc, `las_info_extraction(las_fp,json_combo)`

* function - Displays the classification raster file generated by the .las file and displays the legend, `las_classification_plotWithLegend(las_fp)`

* function - Batch conversion .las point cloud data into DSM and classification raster, `las_info_extraction_combo(las_dirPath,json_combo_)`

* function - Merge multiple rasters into one, `raster_mosaic(dir_path,out_fp,)`

* function - Transfer the function provided by Rasterio that defines the data type with the minimum array size, `get_minimum_int_dtype(values)`

* function -  Gets the projected coordinates of the given raster-crs, `get_crs_raster(raster_fp)`

* function - Transform raster projection,`raster_reprojection(raster_fp,dst_crs,save_path)`

* function - Crop data in .shp format to the extent of a given raster and define a projection as the same as the raster, `clip_shp_withRasterExtent(vector_shp_fp,reference_raster_fp,save_path)`

* function - The interpolation method 'rasterio.fill' is used to complete the missing data,`rasters_interpolation(raster_path,save_path,max_search_distance=400,smoothing_iteration=0)`

* function - Display remote sensing images(one band) using the Earthpy library,`raster_show(raster_fp,title='raster',vmin_vmax=[0.25,0.95],cmap="turbo")`

#### 1.3.3 The python libraries that are being imported


```python
import pdal,os,re,glob
import pandas as pd
import open3d as o3d

import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.plot import plotting_extent
from rasterio.fill import fillnodata

import numpy as np
import earthpy.plot as ep
import geopandas as gpd 
from shapely.geometry import Polygon
import shapely

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib.patches import Rectangle

from tqdm import tqdm
from rasterstats import zonal_stats
from mpl_toolkits.axes_grid1 import make_axes_locatable
```

#### 1.3.4 Reference

-
