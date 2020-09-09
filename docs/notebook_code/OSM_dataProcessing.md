> Created on Fri Dec 27 14/28/05 2019 @author: Richie Bao-caDesign (cadesign.cn) __+updated on Mon Jul  6 19/29/41 2020

## 1.OpenStreetMap（OSM）data processing
[OpenStreetMap(OSM)](https://www.openstreetmap.org/#map=13/41.8679/-87.6569)," Welcome to OpenStreetMap, the project that creates and distributes free geographic data for the world. We started it because most maps you think of as free actually have legal or technical restrictions on their use, holding back people from using them in creative, productive, or unexpected ways. OSM provides the world's roads, paths, cafes, train stations, and other geographic information data, which is a valuable data wealth in the study of urban space.  There are generally two ways to download OSM data by looking at its official website. One is to directly download the display window range of data or input coordinate custom range, but the amount of data downloaded in this way is limited; if the content is too extensive, it can not be downloaded;  The other way is to download directly from the repository. Different resources may be downloaded in different ways according to your own need. Simultaneously, OSM provides many years of historical data, which provides data support for the study of urban spatial changes.The downloaded data downloaded from [Geofabrik](https://download.geofabrik.de/north-america/us.html) repository. Since the analysis's target area is the City of Chicago, the [llinois-latest-free.shp.zip](https://download.geofabrik.de/north-america/us/illinois.html) was downloaded 340MB, and extracted 3.73GB. According to the distribution fo point data, to maintain the continuous area, increase the download [wisconsin-latest.osm.bz2](https://download.geofabrik.de/north-america/us/wisconsin.html) data file is 324MB, 3.72GB after decompression. You can use QGIS to view the data initially. The data layers it contains are 'lines', 'multilinestrings', 'multipolygons', 'other_relations ', and points.

> Below at the same time opens Ilinois and Wisconsin point data, internal red translucent area to [Chicago city administrative scope](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-City/ewy2-6yfk), data from [Chicago Data Portal](https://data.cityofchicago.org/). The outer red dotted line is the actual boundary of experimental data extraction. The smallest black rectangle is used for code debugging to extract small-scale data.

<a href=""><img src="./imgs/4_6.jpg" height="auto" width="500" title="caDesign"></a>

### 1.1 OSM raw data processing
.osm data processing involves merging two areas of data and cutting or cutting separately before joining. It can be based on computer requirements and processing speed, determine the sequence. Here, for cutting .osm data, choosing the cutting method is to use the [osmosis](https://wiki.openstreetmap.org/wiki/Osmosis) command-line tools very suitable for processing large data files, cutting and update data. At the same time, it can refer to [Manipulating Data with Osmosis](https://learnosm.org/en/osm-data/osmosis/). By checking an 'osmosis' case, look for a code using a polygon to extract data: `osmosis --read-xml file="planet-latest.osm" --bounding-polygon file="country2pts.txt" --write-xml file="germany.osm"`. A total of three parameters involved, raw .osm data; the cutting polygon border(.txt data format), to be aware that the polygon that you need to program a transformation code is used for 'osmosis' in polygon format; And the output path.

Visually approximate the range of point aggregation, draw a conventional polygon boundary in QGIS, such as the red dotted line above. First, code a polygon to osmosis format, query the data format as follows:
```
australia_v
first_area
     0.1446693E+03    -0.3826255E+02
     0.1446627E+03    -0.3825661E+02
     0.1446763E+03    -0.3824465E+02
     0.1446813E+03    -0.3824343E+02
     0.1446824E+03    -0.3824484E+02
     0.1446826E+03    -0.3825356E+02
     0.1446876E+03    -0.3825210E+02
     0.1446919E+03    -0.3824719E+02
     0.1447006E+03    -0.3824723E+02
     0.1447042E+03    -0.3825078E+02
     0.1446758E+03    -0.3826229E+02
     0.1446693E+03    -0.3826255E+02
END
second_area
     0.1422436E+03    -0.3839315E+02
     0.1422496E+03    -0.3839070E+02
     0.1422543E+03    -0.3839025E+02
     0.1422574E+03    -0.3839155E+02
     0.1422467E+03    -0.3840065E+02
     0.1422433E+03    -0.3840048E+02
     0.1422420E+03    -0.3839857E+02
     0.1422436E+03    -0.3839315E+02
END
END
```

Format conversion function, call the 'osgeo' class, the class is included in [GDAL](https://pypi.org/project/GDAL/) library, the GDAL is one for the raster and vector open source geospatial data format conversion library, contains a large number of format drive, most of the python geospatial data processing library based on GDAL compiled for the underlying, usually is the most basic library. Geopandas library is based on Fiona and GDAl libraries, making it easy for users to work with geospatial data, and Fiona contains extension modules linked to GDAL. Naturally, there is no restriction on which libraries to use when working with the python geospatial database. The most convenient library for handling geospatial data is Geopandas, but sometimes these libraries do not meet all the requirements, and you need to call GDAL for processing. For GDAL help information, browse the [GDAL/OGR Cookbook!](http://pcjericks.github.io/py-gdalogr-cookbook/index.html), as well as [GDAL documentation](https://gdal.org/).


```python
def shpPolygon2OsmosisTxt(shape_polygon_fp,osmosis_txt_fp): 
    from osgeo import ogr #Osgeo is included in the GDAL library.
    '''
    function-Convert the polygon of the shape to the polygon data format of osmium (.txt) for cropping the .osm map data.
    
    Params:
    shape_polygon_fp - Enter polygon for Shape geographic data format
    osmosis_txt_fp - Output for polygon data format of osmosis .txt
    '''
    driver=ogr.GetDriverByName('ESRI Shapefile') #GDAL is capable of handling many geographic data formats, with the ESRI Shapefile data format driver in place
    infile=driver.Open(shape_polygon_fp) #Open the .shp file
    layer=infile.GetLayer() #Read the layer
    f=open(osmosis_txt_fp,"w") 
    f.write("osmosis polygon\nfirst_area\n")
    
    for feature in layer: 
        feature_shape_polygon=feature.GetGeometryRef() 
        print(feature_shape_polygon) # For the polygon
        firsts_area_linearring=feature_shape_polygon.GetGeometryRef(0) #Polygon contains no nesting but is a separate shape.
        print(firsts_area_linearring) #For the linearRing
        area_vertices=firsts_area_linearring.GetPointCount() #Extract the number of points on the linearRing object
        for vertex in range(area_vertices): #Loop through the point and write the point coordinates to the file
            lon, lat, z=firsts_area_linearring.GetPoint(vertex)  
            f.write("%s  %s\n"%(lon,lat))
    f.write("END\nEND")  
    f.close()   
#Transform the actual experimental boundary
shape_polygon_fp=r'.\data\geoData\OSMBoundary.shp'
osmosis_txt_fp=r'.\data\geoData\OSMBoundary.txt'
shpPolygon2OsmosisTxt(shape_polygon_fp,osmosis_txt_fp)

#Transform code to debug small batch data boundaries
shape_polygon_small_fp=r'.\data\geoData\OSMBoundary_small.shp'
osmosis_txt_small_fp=r'.\data\geoData\OSMBoundary_small.txt'
shpPolygon2OsmosisTxt(shape_polygon_small_fp,osmosis_txt_small_fp)
```

    POLYGON ((-90.0850881031402 40.9968994947319,-90.0850881031402 43.6657936592248,-87.383039973871 43.6657936592248,-87.383039973871 40.9968994947319,-90.0850881031402 40.9968994947319))
    LINEARRING (-90.0850881031402 40.9968994947319,-90.0850881031402 43.6657936592248,-87.383039973871 43.6657936592248,-87.383039973871 40.9968994947319,-90.0850881031402 40.9968994947319)
    POLYGON ((-87.6807286451907 41.8373927809521,-87.6807286451907 41.9214101975252,-87.5941157249019 41.9214101975252,-87.5941157249019 41.8373927809521,-87.6807286451907 41.8373927809521))
    LINEARRING (-87.6807286451907 41.8373927809521,-87.6807286451907 41.9214101975252,-87.5941157249019 41.9214101975252,-87.5941157249019 41.8373927809521,-87.6807286451907 41.8373927809521)
    

After performing the conversion, write the .txt format file to the real experimental edge polygon as follows:
```
osmosis polygon
first_area
-90.08508810314017  40.99689949473193
-90.08508810314017  43.66579365922478
-87.38303997387102  43.66579365922478
-87.38303997387102  40.99689949473193
-90.08508810314017  40.99689949473193
END
END
```

For debugging small batch data extraction .txt boundary:
```
osmosis polygon
first_area
-87.68072864519071  41.83739278095207
-87.68072864519071  41.92141019752525
-87.59411572490187  41.92141019752525
-87.59411572490187  41.83739278095207
-87.68072864519071  41.83739278095207
END
END
```

Osmosis also provides a plurality of .osm geospatial data merging tools, with a sample merging code of `osmosis --rx 1.osm --rx 2.osm --rx 3.osm --merge --wx merged.osm`. First, perform the merging; osmosis combined code for this test is `osmosis --rx "F:/GitHubBigData/illinois-latest.osm" --rx "F:/GitHubBigData/wisconsin-latest.osm"--merge --wx "F:/GitHubBigData/illinois-wisconsin.osm"`，The integrated file size is 7.57GB. Then execute the cut order as `osmosis --read-xml file="F:\GitHubBigData\illinois-wisconsin.osm" --bounding-polygon file="C:\Users\richi\omen-richiebao\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\data\geoData\OSMBoundary.txt" --write-xml file="F:\GitHubBigData\osm_clip.osm"`，The cropped file size is 3.80GB. QGIS can be used to check whether the data has been combined and cut as expected. The small data extraction and cutting tool for code debugging is `osmosis --read-xml file="F:\GitHubBigData\illinois-wisconsin.osm" --bounding-polygon file="C:\Users\richi\omen-richiebao\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\data\geoData\OSMBoundary_small.txt" --write-xml file="F:\GitHubBigData\osm_small_clip.osm"`.

> Osmosis command line, executed in Windows system command line terminal, recommended running in [Windows PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7) terminal.

<a href=""><img src="./imgs/4_5.jpg" height="auto" width="auto" title="caDesign"></a>

OSM uses tags attached to fundamental data structures to represent physical features on the ground, such as roads or buildings. Open the attribute table to see to which tags each element belongs. For the specific tag, you can view [Map Features](https://wiki.openstreetmap.org/wiki/Map_Features); the following list only the classification of the main tags:

| idex   |      Level 1 tag      |  Level 2 tag |
|----------|:-------------|------:|
| 1 |   Aerialway |  |
| 2 |     Aeroway   |   |
| 3 |  Amenity |     Sustenance,Education,  Transportation, Financial, Healthcare, Entertainment, Arts & Culture, Others|
|4| Barrier| Linear barriers, Access control on highways|
| 5 |  Boundary |Attributes   |
| 6| Building |  Accommodation, Commercial, Religious,Civic/Amenity, Agricultural/Plant production, Sports,Storage,Cars, Power/Technical buildings,Other Buildings, Additional Attributes|
| 7 | Craft |   |
| 8 | Emergency | Medical Rescue,Firefighters, Lifeguards,  Assembly point,Other Structure |
| 9 |  Geological |   |
| 10 | Highway |  Roads, Link roads, Special road types, Paths,Lifecycle, Attributes,Other highway features|
| 11 | Historic |   |
| 12 | Landuse | Common Landuse Key Values - Developed land, Common Landuse Key Values - Rural and agricultural land, Other Landuse Key Values |
| 13 | Leisure |   |
| 14 |  Man_made |   |
| 15 | Military |   |
| 16 | Natural |  Vegetation or surface related, Water related, Landform related |
| 17 |Office  |   |
| 18 | Place | Administratively declared places,   Populated settlements, urban,Populated settlements, urban and rural, Other places|
| 19 |  Power |  Public Transport |
| 20 | Public Transport |  |
| 21 | Railway |  Tracks, Additional features, Stations and Stops, Other railways|
| 22 | Route |   |
| 23 | Shop | Food, beverages,  General store, department store, mall,  Clothing, shoes, accessories, Discount store, charity, Health and beauty, Do-it-yourself, household, building materials, gardening, Furniture and interior, Electronics, Outdoors and sport, vehicles, Art, music, hobbies,Stationery, gifts, books, newspapers, Others|
| 24 | Sport  |   |
| 25 |  Telecom |   |
| 26 |Tourism  |   |
| 27 |  Waterway | Natural watercourses, Man-made waterways, Facilities,Barriers on waterways,Other features on waterways|
| Additional properties |  |   |
| 1 | Addresses | Tags for individual houses, For countries using hamlet, subdistrict, district, province, state, Tags for interpolation ways |
| 2 | Annotation  |   |
| 3 | Name |   |
| 4 | Properties  |   |
| 5 | References  |   |
| 6 | Restrictions  |   |


> Osmosis tool is developed by OSM and members of its open source community based on [osmcode.org](https://osmcode.org/).

### 1.2 Read and convert .osm data
Read .osm data in python still use the tools [pyosmium](https://docs.osmcode.org/pyosmium/latest/) provided by [osmcode.org](https://osmcode.org/), pyosmium is processing the OSM files in different formats, the kernel for c++ library of osmium can effectively deal with OSM data rapidly. The above processed .osm data file osm_clip.osm is 3.80GB. If the code is written with more massive data initially, the time cost may be higher. It can be written and debugged with less data, and the large file data to be analyzed can be used after the desired effect is achieved.

To read and write .osm data, you need to understand the OSM data structure so that you can extract the required values. [elements](https://wiki.openstreetmap.org/wiki/Elements) is the OSM in the physical world, an essential part of the conceptual data model, including nodes, paths or ways, relationship, as well as the tags.

| elements   |     icon     |  explanation |Shape, geospatial data（vector）|
|----------|:-------------|------:|------:|
| node |   <a href=""><img src="./imgs/30px-Osm_element_node.svg.png" height="auto" width="auto" title="caDesign"></a> |A geospatial point defined by latitude and longitude coordinates  |point|
| way |    <a href=""><img src="./imgs/30px-Osm_element_way.svg.png" height="auto" width="auto" title="caDesign"></a><a href=""><img src="./imgs/30px-Osm_element_closedway.svg.png" height="auto" width="auto" title="caDesign"></a><a href=""><img src="./imgs/30px-Osm_element_area.svg.png" height="auto" width="auto" title="caDesign"></a>   | A path  and a closed region consisting of (20-2000) points, containing 
 open way, closed way, and area  |polyline,polygon|
| relation |     <a href=""><img src="./imgs/30px-Osm_element_relation.svg.png" height="auto" width="auto" title="caDesign"></a>   | A multi-purpose data structure that records relationships between two or more elements, which can have different meanings defined by the corresponding tags ||
| tag|  <a href=""><img src="./imgs/30px-Osm_element_tag.svg.png" height="auto" width="auto" title="caDesign"></a>   | nodes, ways, and relations can all be represented by tags that describe their meaning. A tag consist of a key: value, and the key must be unique  | field|

* example

**node**
```html
<node id="25496583" lat="51.5173639" lon="-0.140043" version="1" changeset="203496" user="80n" uid="1238" visible="true" timestamp="2007-01-28T11:40:26Z">
    <tag k="highway" v="traffic_signals"/>
</node>
```

**way**

simple way
```html
  <way id="5090250" visible="true" timestamp="2009-01-19T19:07:25Z" version="8" changeset="816806" user="Blumpsy" uid="64226">
    <nd ref="822403"/>
    <nd ref="21533912"/>
    <nd ref="821601"/>
    <nd ref="21533910"/>
    <nd ref="135791608"/>
    <nd ref="333725784"/>
    <nd ref="333725781"/>
    <nd ref="333725774"/>
    <nd ref="333725776"/>
    <nd ref="823771"/>
    <tag k="highway" v="residential"/>
    <tag k="name" v="Clipstone Street"/>
    <tag k="oneway" v="yes"/>
  </way>
```

multipolygon area

<img src="./imgs/300px-Multipolygon_Illustration_2.svg.png" height="auto" width="auto" title="caDesign"><img src="./imgs/300px-Multipolygon_Illustration_1b.svg.png" height="auto" width="auto" title="caDesign">

```html
  <relation id="12" timestamp="2008-12-21T19:31:43Z" user="kevjs1982" uid="84075">
    <member type="way" ref="2878061" role="outer"/> <!-- picture ref="1" -->
    <member type="way" ref="8125153" role="inner"/> <!-- picture ref="2" -->
    <member type="way" ref="8125154" role="inner"/> <!-- picture ref="3" -->

    <member type="way" ref="3811966" role=""/> <!-- empty role produces
        a warning; avoid this; most software works around it by computing
        a role, which is more expensive than having one set explicitly;
        not shown in the sample pictures to the right -->

    <tag k="type" v="multipolygon"/>
  </relation>
```




* Attributes of elements

| name   | value type         |  explanation |
|----------|:-------------|------:|
| id |integer (64-bit)   | Used to represent elements |
| user |character string   | The user name of the object is last modified|
| uid | integer | The user ID of the object is last modified|
| timestamp | W3C standard date and time  | Last modified time |
|visible  | "true" or "false"  |Whether an object in the database has been deleted |
|version  | integer  | Version control |
|changeset  | integer  |The changeset number used when an object is created or updated |

Learn the basic data types, structures, and properties of OSM, inheriting the osmium's class_SimpleHandler, passing in the .osm file using .apply_file method, and define the type of element to be extracted, and give the attributes of the element type to extract the corresponding attribute values. In geospatial data analysis, commonly, key features include:  last time of element modification(.timestamp), tags(tags,<tag.k,tag.v>), generated geometry objects(geometry<point,linestring,multipolygon>). The following functions extract the properties and geometric objects of node, way(area) respectively, convert them into the data format of GeoDataFrame, and save them as `GPKG' data format for future calls, especially for large quantities of data. Since a large amount of data is involved, you can call in the 'datatime' module, observe the time being spent, and help debug the code.


```python
import osmium as osm
import pandas as pd
import datetime
import shapely.wkb as wkblib
wkbfab=osm.geom.WKBFactory()

class osmHandler(osm.SimpleHandler):    
    '''
    class-Read the .osm data by inheriting the osmium class osmium.SimpleHandler
    '''
    
    def __init__(self):
        osm.SimpleHandler.__init__(self)
        self.osm_node=[]
        self.osm_way=[]
        self.osm_area=[]
        
    def node(self,n):
        wkb=wkbfab.create_point(n)
        point=wkblib.loads(wkb,hex=True)
        self.osm_node.append([
            'node',
            point,
            n.id,
            n.version,
            n.visible,
            pd.Timestamp(n.timestamp),
            n.uid,
            n.user,
            n.changeset,
            len(n.tags),
            {tag.k:tag.v for tag in n.tags},
            ])

    def way(self,w):     
        try:
            wkb=wkbfab.create_linestring(w)
            linestring=wkblib.loads(wkb, hex=True)
            self.osm_way.append([
                'way',
                linestring,
                w.id,
                w.version,
                w.visible,
                pd.Timestamp(w.timestamp),
                w.uid,
                w.user,
                w.changeset,
                len(w.tags),
                {tag.k:tag.v for tag in w.tags}, 
                ])
        except:
            pass
        
    def area(self,a):     
        try:
            wkb=wkbfab.create_multipolygon(a)
            multipolygon=wkblib.loads(wkb, hex=True)
            self.osm_area.append([
                'area',
                multipolygon,
                a.id,
                a.version,
                a.visible,
                pd.Timestamp(a.timestamp),
                a.uid,
                a.user,
                a.changeset,
                len(a.tags),
                {tag.k:tag.v for tag in a.tags}, 
                ])
        except:
            pass      
        
a_T=datetime.datetime.now()
print("start time:",a_T)
#osm_Chicago_fp=r"F:\GitHubBigData\osm_small_clip.osm" #The .osm data path to read, debug the code with the small range of extracted data
osm_Chicago_fp=r"F:\GitHubBigData\osm_clip.osm" #After debugging with small-batch data, calculate the actual experimental data

osm_handler=osmHandler() #Instantiate the class osmHandler()
osm_handler.apply_file(osm_Chicago_fp,locations=True) #Call the apply_file method of the class osmium.SimpleHandler
b_T=datetime.datetime.now()
print("end time:",b_T)
duration=(b_T-a_T).seconds/60
print("Total time spend:%.2f minutes"%duration)
```

    start time: 2020-07-08 16:38:01.285606
    end time: 2020-07-08 17:12:23.316155
    Total time spend:34.37 minutes
    

When all OSM element data is read, the save function is defined. If the data is a small batch, it can usually be saved together. However, this experiment has 3.80GB of data, so it takes more time to convert data into the GeoDataFrame data format and then save it. So save OSM elements one by one. Plus, note that when debugging in small batches, the size of the node file saved as GeoJSON is 104MB, while the size of the file saved as GPKG is only 52.3MB, so the latter is selected here for the experimental data.


```python
def save_osm(osm_handler,osm_type,save_path=r"./data/",fileType="GPKG"):
    a_T=datetime.datetime.now()
    print("start time:",a_T)
    import geopandas as gpd
    import os
    import datetime
    '''
    function-Save the OSM data one by one, depending on the condition.（node, way and area）
    
    Paras:
    osm_handler -  OSM returns node, way, and area data
    osm_type - Type of OSM element to save
    save_path - Saved path
    fileType - Type of data saved，shp, GeoJSON, GPKG
    '''
    def duration(a_T):
        b_T=datetime.datetime.now()
        print("end time:",b_T)
        duration=(b_T-a_T).seconds/60
        print("Total time spend:%.2f minutes"%duration)
        
    def save_gdf(osm_node_gdf,fileType,osm_type):
        if fileType=="GeoJSON":
            osm_node_gdf.to_file(os.path.join(save_path,"osm_%s.geojson"%osm_type),driver='GeoJSON')
        elif fileType=="GPKG":
            osm_node_gdf.to_file(os.path.join(save_path,"osm_%s.gpkg"%osm_type),driver='GPKG')
        elif fileType=="shp":
            osm_node_gdf.to_file(os.path.join(save_path,"osm_%s.shp"%osm_type))

    crs={'init': 'epsg:4326'} #Coordinate system configuration, reference：https://spatialreference.org/        
    osm_columns=['type','geometry','id','version','visible','ts','uid','user','changeet','tagLen','tags']
    if osm_type=="node":
        osm_node_gdf=gpd.GeoDataFrame(osm_handler.osm_node,columns=osm_columns,crs=crs)
        save_gdf(osm_node_gdf,fileType,osm_type)
        duration(a_T)
        return osm_node_gdf

    elif osm_type=="way":
        osm_way_gdf=gpd.GeoDataFrame(osm_handler.osm_way,columns=osm_columns,crs=crs)
        save_gdf(osm_way_gdf,fileType,osm_type)
        duration(a_T)
        return osm_way_gdf
        
    elif osm_type=="area":
        osm_area_gdf=gpd.GeoDataFrame(osm_handler.osm_area,columns=osm_columns,crs=crs)
        save_gdf(osm_area_gdf,fileType,osm_type)
        duration(a_T)
        return osm_area_gdf
```


```python
node_gdf=save_osm(osm_handler,osm_type="node",save_path=r"./data/",fileType="GPKG")
```

    start time: 2020-07-08 17:13:45.751609
    end time: 2020-07-08 18:40:34.442122
    Total time spend:86.80 minutes
    


```python
way_gdf=save_osm(osm_handler,osm_type="way",save_path=r"./data/",fileType="GPKG")
```

    start time: 2020-07-08 18:45:26.586573
    end time: 2020-07-08 18:57:58.356928
    Total time spend:12.52 minutes
    


```python
area_gdf=save_osm(osm_handler,osm_type="area",save_path=r"./data/",fileType="GPKG")
```

    start time: 2020-07-08 18:59:12.339119
    end time: 2020-07-08 19:09:24.169272
    Total time spend:10.18 minutes
    

In the stored procedure, for this part of the experimental data, the node element's storage time is about 87 minutes, and the GPKG file size is 3.10GB, while the way and areal elements have relatively short storage time and small storage file size. Since the GeoDataFrame geospatial data format has been converted, you can go directly to .plot() to see the data distribution and make a preliminary decision as to whether the data has been read and converted correctly. The following code is used to test the read duration, with the way and area taking a shorter time and the node element taking a longer time. 


```python
def start_time():
    import datetime
    '''
    function-Calculate the current time
    '''
    start_time=datetime.datetime.now()
    print("start time:",start_time)
    return start_time

def duration(start_time):
    import datetime
    '''
    function-Calculate the duration
    
    Paras:
    start_time - The start time
    '''
    end_time=datetime.datetime.now()
    print("end time:",end_time)
    duration=(end_time-start_time).seconds/60
    print("Total time spend:%.2f minutes"%duration)
```


```python
import geopandas as gpd
start_time=start_time()
read_way_gdf=gpd.read_file("./data/osm_way.gpkg")
duration(start_time)
```

    start time: 2020-07-08 20:30:02.328322
    end time: 2020-07-08 20:31:51.139907
    Total time spend:1.80 minutes
    


```python
read_way_gdf.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x2508dff6c48>



<img src="./imgs/4_1.png" height="auto" width="500" title="caDesign">

```python
del read_way_gdf #If memory is limited, del can be used to remove variables that are no longer used, thereby saving memory.
```


```python
start_time=start_time()
read_area_gdf=gpd.read_file("./data/osm_area.gpkg")
duration(start_time)
```

    start time: 2020-07-08 20:13:53.521419
    end time: 2020-07-08 20:16:28.547289
    Total time spend:2.58 minutes
    


```python
read_area_gdf.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x250ba44aa48>




<img src="./imgs/4_2.png" height="auto" width="500" title="caDesign">



```python
start_time=start_time()
read_node_gdf=gpd.read_file("./data/osm_node.gpkg")
duration(start_time)
```

    start time: 2020-07-08 20:32:25.929397
    end time: 2020-07-08 21:10:42.912870
    Total time spend:38.27 minutes
    


```python
read_node_gdf.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x250ba0ed308>




<img src="./imgs/4_3.png" height="auto" width="500" title="caDesign">


### 1.3 key point
#### 1.3.1 data processing technique

* .osm data processing library，The [osmosis](https://wiki.openstreetmap.org/wiki/Osmosis) command-line tool provided by [osmcode.org](https://osmcode.org/) handles raw data, as well as the [pyosmium](https://docs.osmcode.org/pyosmium/latest/) library, which supports processing in the python language.

* GDAL, python's most basic library for handling geospatial data

* The datatime library is used to get the time and calculate the running time of the program

* The method of del variable is used to delete the variables not used to save the memory space.

#### 1.3.2 The newly created function tool

* function-Convert the polygon of the shape to the polygon data format of osmium (.txt) for cropping the .osm map data.`shpPolygon2OsmosisTxt(shape_polygon_fp,osmosis_txt_fp)`

* class-Read the .osm data by inheriting the osmium class osmium.SimpleHandler, `osmHandler(osm.SimpleHandler)`

* function-Save the OSM data one by one, depending on the condition.（node, way and area）,`save_osm(osm_handler,osm_type,save_path=r"./data/",fileType="GPKG")`

* function-Calculate the current time, `start_time()`

* function-Calculate the duration, `duration(start_time)`

#### 1.3.3 The python libraries that are being imported


```python
from osgeo import ogr
import osmium as osm
import pandas as pd
import datetime
import shapely.wkb as wkblib
import geopandas as gpd
import os
```