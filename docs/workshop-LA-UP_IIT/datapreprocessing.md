> Created on Fri Feb 26 13\12\28 2021 @author: richie bao -workshop-LA-UP_IIT

## 1. Data Preprocessing

### 1.1 Define geographic projection and data file paths


```python
nanjing_epsg=32650 #Nanjing
data_dic={
    'road_network':r'.\data\GIS\road Network Data OF Nanjing On 20190716.kml',
    'qingliangMountain_boundary':r'./data/GIS/QingliangMountain_boundary.kml',
    'building_footprint':r'./data/GIS/Nanjing Building footprint Data/NanjingBuildingfootprintData.shp',
    'bus_routes':r'./data/GIS/SHP data of Nanjing bus route and stations in December 2020/busRouteStations_20201218135814.shp',
    'bus_stations':r'./data/GIS/SHP data of Nanjing bus route and stations in December 2020/busRouteStations_20201218135812.shp',
    'subway_lines':r'./data/GIS/SHP of Nanjing subway station and line on 2020/SHP of Nanjing subway station and line on 2020 (2).shp',
    'subway_stations':r'./data/GIS/SHP of Nanjing subway station and line on 2020/SHP of Nanjing subway station and line on 2020.shp',
    'population':r'./data/GIS/SHP of population distribution in Nanjing in 2020/SHP of population distribution in Nanjing in 2020.shp',
    'taxi':r'./data/GIS/Nanjing taxi data in 2016',
    'POI':r"./data/GIS/Nanjing POI 201912.csv",
    'microblog':r'./data/GIS/Najing Metro Weibo publish.db',
    'bike_sharing':r'./data/GIS/One hundred thousand shared bikes.xls',
    'sentinel_2':r'C:\Users\richi\omen_richiebao\omen_IIIT\workshop_LA_UP_iit\data\RS\S2B_MSIL2A_20200819T024549_N0214_R132_T50SPA_20200819T045147.SAFE',
    }
```

### 1.2 define data analysis boundaries

* Method 1, given the bottom left, upper right points define the boundary.


```python
def boundary_angularPts(bottom_left_lon,bottom_left_lat,top_right_lon,top_right_lat):
    bottom_left=(bottom_left_lon,bottom_left_lat)
    top_right=(top_right_lon,top_right_lat)
    boundary=[(bottom_left[0],bottom_left[1]),(top_right[0],bottom_left[1]),(top_right[0], top_right[1]),(bottom_left[0], top_right[1])]
    return boundary
```

* Method 2, Draw .kml boundary under Google Earth and establish analysis buffer. There are two modes of point buffer and boundary buffer.


```python
def boundary_buffer_centroidCircle(kml_extent,proj_epsg,bounadry_type='buffer_circle',buffer_distance=1000):
    import pyproj
    from shapely.ops import transform
    from shapely.geometry import Point,LinearRing,Polygon
    import geopandas as gpd
    
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    boundary_gdf=gpd.read_file(kml_extent,driver='KML')
    # print(boundary_gdf)
    
    wgs84=pyproj.CRS('EPSG:4326')
    utm=pyproj.CRS('EPSG:{}'.format(proj_epsg))
    project=pyproj.Transformer.from_crs(wgs84, utm, always_xy=True).transform

    boundary_proj=transform(project,boundary_gdf.geometry.values[0])    
    
    if bounadry_type=='buffer_circle':
        b_centroid=boundary_proj.centroid
        b_centroid_buffer=b_centroid.buffer(buffer_distance)
        c_area=[b_centroid_buffer.area]
        gpd.GeoDataFrame({'area': c_area,'geometry':b_centroid_buffer},crs=utm).to_crs(wgs84).to_file('./data/GIS/b_centroid_buffer.shp')        
        return b_centroid_buffer
    
    elif bounadry_type=='buffer_offset':
        boundary_=Polygon(boundary_proj.exterior.coords)
        LR_buffer=boundary_.buffer(buffer_distance,join_style=1).difference(boundary_)  #LinearRing  
        # LR_buffer=Polygon(boundary_proj.exterior.coords)
        LR_area=[LR_buffer.area]
        gpd.GeoDataFrame({'area': LR_area,'geometry':LR_buffer},crs=utm).to_crs(wgs84).to_file('./data/GIS/LR_buffer.shp')  
        return LR_buffer
```


```python
kml_extent=data_dic['qingliangMountain_boundary']
boudnary_polygon=boundary_buffer_centroidCircle(kml_extent,nanjing_epsg,bounadry_type='buffer_circle',buffer_distance=5000) #'buffer_circle';'buffer_offset'
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\geopandas\geodataframe.py:422: RuntimeWarning: Sequential read of iterator was interrupted. Resetting iterator. This can negatively impact the performance.
      for feature in features_lst:
    


```python
boudnary_polygon
```




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_18.png" height='auto' width='auto' title="caDesign"></a>  
    



### 1.3 convert .kml data to GeoDataFrame


```python
def kml2gdf(fn,epsg=None,boundary=None): 
    import pandas as pd
    import geopandas as gpd
    from tqdm import tqdm
    import fiona,io
    
    # Enable fiona driver
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    kml_gdf=gpd.GeoDataFrame()
    for layer in tqdm(fiona.listlayers(fn)):
        src=fiona.open(fn, layer=layer)
        meta = src.meta
        meta['driver'] = 'KML'        
        with io.BytesIO() as buffer:
            with fiona.open(buffer, 'w', **meta) as dst:            
                for i, feature in enumerate(src):
                    if len(feature['geometry']['coordinates']) > 1:
                        # print(feature['geometry']['coordinates'])
                        dst.write(feature)
                        # break
            buffer.seek(0)
            one_layer=gpd.read_file(buffer,driver='KML')
            one_layer['group']=layer
            kml_gdf=kml_gdf.append(one_layer,ignore_index=True)            
    
    # crs={'init': 'epsg:4326'}
    if epsg is not None:
        kml_gdf_proj=kml_gdf.to_crs(epsg=epsg)

    if boundary:
        kml_gdf_proj['mask']=kml_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        kml_gdf_proj.query('mask',inplace=True)        

    return kml_gdf_proj
```

* conversion of the road network


```python
road_gdf=kml2gdf(data_dic['road_network'],epsg=nanjing_epsg,boundary=boudnary_polygon)
road_gdf.plot()
```

      0%|          | 0/28 [00:00<?, ?it/s]C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\ipykernel_launcher.py:16: RuntimeWarning: Sequential read of iterator was interrupted. Resetting iterator. This can negatively impact the performance.
      app.launch_new_instance()
    100%|██████████| 28/28 [00:08<00:00,  3.12it/s]
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_04.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
road_gdf
```




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
      <th>Name</th>
      <th>Description</th>
      <th>geometry</th>
      <th>group</th>
      <th>mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td></td>
      <td>"标识码":"48f23105-7d08-467c-aa71-3f57457028e5" "...</td>
      <td>MULTILINESTRING Z ((666266.776 3546407.125 0.0...</td>
      <td>002 南京市第二次地名普查界线</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>"标识码":"48f23105-7d08-467c-aa71-3f57457028e5" "...</td>
      <td>MULTILINESTRING Z ((666266.776 3546407.125 0.0...</td>
      <td>002 南京市第二次地名普查界线</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>"标识码":"c129aad9-baed-4bd3-8cf1-e1ed4d3e23b9" "...</td>
      <td>LINESTRING Z (666130.351 3546636.914 0.000, 66...</td>
      <td>002 南京市第二次地名普查界线</td>
      <td>True</td>
    </tr>
    <tr>
      <th>29</th>
      <td></td>
      <td>"标识码":"c129aad9-baed-4bd3-8cf1-e1ed4d3e23b9" "...</td>
      <td>LINESTRING Z (666130.351 3546636.914 0.000, 66...</td>
      <td>002 南京市第二次地名普查界线</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39</th>
      <td></td>
      <td>"标识码":"48f23105-7d08-467c-aa71-3f57457028e5" "...</td>
      <td>MULTILINESTRING Z ((666266.776 3546407.125 0.0...</td>
      <td>秦淮</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2866</th>
      <td>双塘路</td>
      <td>官方批复或地名录: 《南京市地名查询系统》: 《南京市地名大全》:位于内秦淮河南。东起上浮桥...</td>
      <td>LINESTRING Z (666794.663 3544709.560 0.000, 66...</td>
      <td>320115江宁区</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3103</th>
      <td>太平巷</td>
      <td>官方批复或地名录: 《南京市地名查询系统》: 《南京市地名大全》:位于郑和公园北侧。东起长白...</td>
      <td>LINESTRING Z (668778.168 3545717.944 0.000, 66...</td>
      <td>320115江宁区</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3153</th>
      <td>三茅巷</td>
      <td>官方批复或地名录: 《南京市地名查询系统》: 《南京市地名大全》:位于莫愁路东侧。东起王府大...</td>
      <td>LINESTRING Z (667176.277 3546357.798 0.000, 66...</td>
      <td>320115江宁区</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4057</th>
      <td>厚德路</td>
      <td>官方批复或地名录:宁地名办[2013]109号:因该道路连接主干道与殡仪馆，殡仪馆一直以厚德...</td>
      <td>LINESTRING Z (667119.486 3550348.583 0.000, 66...</td>
      <td>320116六合区</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4100</th>
      <td>永宁街</td>
      <td>官方批复或地名录:六合县地名录:无《地名录》：明时即有此街,因处滁河北岸,地势低洼,常遭水患...</td>
      <td>LINESTRING Z (663827.040 3551662.414 0.000, 66...</td>
      <td>320116六合区</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>759 rows × 5 columns</p>
</div>



### 1.4 convert .shp data to GeoDataFrame


```python
def  shp2gdf(fn,epsg=None,boundary=None,encoding='utf-8'):
    import geopandas as gpd
    
    shp_gdf=gpd.read_file(fn,encoding=encoding)
    print('original data info:{}'.format(shp_gdf.shape))
    shp_gdf.dropna(how='all',axis=1,inplace=True)
    print('dropna-how=all,result:{}'.format(shp_gdf.shape))
    shp_gdf.dropna(inplace=True)
    print('dropna-several rows,result:{}'.format(shp_gdf.shape))
    # print(shp_gdf)
    if epsg is not None:
        shp_gdf_proj=shp_gdf.to_crs(epsg=epsg)
    if boundary:
        shp_gdf_proj['mask']=shp_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        shp_gdf_proj.query('mask',inplace=True)        
    
    return shp_gdf_proj
```

* conversion of the building footprint


```python
buildingFootprint=shp2gdf(data_dic['building_footprint'],epsg=nanjing_epsg,boundary=boudnary_polygon)
buildingFootprint.plot(column='Floor',cmap='terrain')
```

    original data info:(218958, 3)
    dropna-how=all,result:(218958, 3)
    dropna-several rows,result:(217969, 3)
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_05.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
buildingFootprint
```




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
      <th>Id</th>
      <th>Floor</th>
      <th>geometry</th>
      <th>mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10610</th>
      <td>0</td>
      <td>2</td>
      <td>POLYGON ((660695.057 3546708.295, 660703.228 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10611</th>
      <td>0</td>
      <td>2</td>
      <td>POLYGON ((660702.909 3546723.535, 660702.973 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10612</th>
      <td>0</td>
      <td>2</td>
      <td>POLYGON ((660689.149 3546698.125, 660703.308 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10613</th>
      <td>0</td>
      <td>2</td>
      <td>POLYGON ((660604.390 3546556.727, 660630.941 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>10614</th>
      <td>0</td>
      <td>2</td>
      <td>POLYGON ((660697.614 3546546.940, 660705.785 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>159948</th>
      <td>0</td>
      <td>1</td>
      <td>POLYGON ((670326.286 3546847.266, 670335.422 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>159952</th>
      <td>0</td>
      <td>1</td>
      <td>POLYGON ((670324.290 3546784.683, 670336.488 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>159959</th>
      <td>0</td>
      <td>1</td>
      <td>POLYGON ((670337.706 3546832.348, 670357.026 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>159960</th>
      <td>0</td>
      <td>1</td>
      <td>POLYGON ((670356.282 3546814.538, 670359.345 3...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>159976</th>
      <td>0</td>
      <td>2</td>
      <td>POLYGON ((670307.883 3546491.815, 670313.990 3...</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>91468 rows × 4 columns</p>
</div>



* converstion of the bus routes


```python
bus_routes=shp2gdf(data_dic['bus_routes'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
bus_routes.plot()
```

    original data info:(1304, 9)
    dropna-how=all,result:(1304, 8)
    dropna-several rows,result:(1304, 8)
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_06.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
bus_routes
```




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
      <th>LineName</th>
      <th>LineUid</th>
      <th>StartTime</th>
      <th>EndTime</th>
      <th>WorkTimeDe</th>
      <th>Direction</th>
      <th>Price</th>
      <th>geometry</th>
      <th>mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>239</th>
      <td>20路高峰区间(莫愁湖公园西门-鼓楼公园)</td>
      <td>40e351a4cb63d2eb373a7111</td>
      <td>07:30</td>
      <td>18:30</td>
      <td>周一至周五 早07:30-09:30 晚16:30-18:30</td>
      <td>鼓楼公园方向</td>
      <td>2</td>
      <td>LINESTRING (665444.340 3546009.017, 665436.950...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>240</th>
      <td>20路高峰区间(鼓楼公园-莫愁新寓)</td>
      <td>8cc70507af100dbb86420e68</td>
      <td>08:00</td>
      <td>19:00</td>
      <td>周一至周五 早08:00-10:00 晚17:00-19:00</td>
      <td>莫愁新寓方向</td>
      <td>2</td>
      <td>LINESTRING (667641.986 3548597.033, 667604.217...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>294</th>
      <td>302路(中华门城堡-古平岗)</td>
      <td>672f3ec0bdf8bae8b7f42f1b</td>
      <td>06:30</td>
      <td>21:00</td>
      <td>周一至周五 06:30-21:00  周六日 06:30-21:00</td>
      <td>古平岗方向</td>
      <td>2</td>
      <td>LINESTRING (667850.894 3543567.094, 667842.051...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>295</th>
      <td>302路(古平岗-中华门城堡)</td>
      <td>24a851ff3ca68ebddb2f281b</td>
      <td>06:00</td>
      <td>20:30</td>
      <td>周一至周五 06:00-20:30  周六日 06:00-20:30</td>
      <td>中华门城堡方向</td>
      <td>2</td>
      <td>LINESTRING (665348.199 3549954.698, 665347.653...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>296</th>
      <td>303路(金贸大街·建宁路-万达广场南)</td>
      <td>4897d0a108f3e266aaf7291b</td>
      <td>06:00</td>
      <td>21:30</td>
      <td>周一至周五 06:00-21:30  周六日 06:00-21:30</td>
      <td>万达广场南方向</td>
      <td>2</td>
      <td>LINESTRING (667473.334 3551821.902, 667471.661...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>297</th>
      <td>303路(万达广场南-金贸大街·建宁路)</td>
      <td>c8c9e4f4642893beafb22a1b</td>
      <td>06:00</td>
      <td>21:30</td>
      <td>周一至周五 06:00-21:30  周六日 06:00-21:30</td>
      <td>金贸大街·建宁路方向</td>
      <td>2</td>
      <td>LINESTRING (663484.127 3545162.628, 663481.818...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>338</th>
      <td>317路(清江南路(越洋国际)-西街)</td>
      <td>8b4e8621ffd171b4e407385c</td>
      <td>06:00</td>
      <td>20:20</td>
      <td>周一至周五 06:00-20:20  周六日 06:00-20:20</td>
      <td>西街方向</td>
      <td>2</td>
      <td>LINESTRING (662610.955 3545518.771, 662645.727...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>339</th>
      <td>317路(西街-清江南路)</td>
      <td>8b8c0cd6f7fadd4ea4da395c</td>
      <td>06:40</td>
      <td>21:00</td>
      <td>周一至周五 06:40-21:00  周六日 06:40-21:00</td>
      <td>清江南路方向</td>
      <td>2</td>
      <td>LINESTRING (667314.299 3543180.433, 667333.097...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>344</th>
      <td>31路(建康路-中山码头)</td>
      <td>979817573370194b9ce3211b</td>
      <td>05:30</td>
      <td>22:15</td>
      <td>周一至周五 05:30-22:15  周六日 05:30-22:15</td>
      <td>中山码头方向</td>
      <td>2</td>
      <td>LINESTRING (668941.267 3544795.714, 668954.137...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>345</th>
      <td>31路(中山码头-建康路)</td>
      <td>41e096f0fb50b53e2002201b</td>
      <td>05:15</td>
      <td>22:15</td>
      <td>周一至周五 05:15-22:15  周六日 05:15-22:15</td>
      <td>建康路方向</td>
      <td>2</td>
      <td>LINESTRING (663193.969 3551686.724, 663260.131...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>421</th>
      <td>3路(随家仓(新街口方向)-随家仓)</td>
      <td>a9558f191cd504ab461a323e</td>
      <td>05:00</td>
      <td>22:30</td>
      <td>05:00-22:30</td>
      <td>随家仓方向</td>
      <td>2</td>
      <td>LINESTRING (666825.868 3547735.096, 666821.007...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>422</th>
      <td>3路(随家仓(湖南路方向)-随家仓)</td>
      <td>b9574fac23c24f3b43ab9a1a</td>
      <td>05:00</td>
      <td>22:30</td>
      <td>05:00-22:30</td>
      <td>随家仓方向</td>
      <td>2</td>
      <td>LINESTRING (666825.503 3547729.000, 666824.732...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>83路(新模范马路东(市公交集团)-白鹭花园)</td>
      <td>6951a2ec3431f54d057ce8f4</td>
      <td>05:20</td>
      <td>22:00</td>
      <td>周一至周五 05:20-22:00  周六日 05:20-22:00</td>
      <td>白鹭花园方向</td>
      <td>2</td>
      <td>LINESTRING (667718.167 3550652.897, 667498.690...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1018</th>
      <td>83路(白鹭花园-新模范马路东(市公交集团))</td>
      <td>5ae882dce1b94802705e80f4</td>
      <td>06:00</td>
      <td>22:00</td>
      <td>周一至周五 06:00-22:00  周六日 06:00-22:00</td>
      <td>新模范马路东(市公交集团)方向</td>
      <td>2</td>
      <td>LINESTRING (664340.163 3545073.949, 664390.200...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>9路(宁工新寓-总统府)</td>
      <td>735e1202f26a751cbca2401b</td>
      <td>05:30</td>
      <td>21:20</td>
      <td>周一至周五 05:30-21:20  周六日 05:30-21:20</td>
      <td>总统府方向</td>
      <td>2</td>
      <td>LINESTRING (663084.492 3548352.084, 663134.425...</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1187</th>
      <td>9路(总统府-宁工新寓)</td>
      <td>9e7d14291a86012e306212d4</td>
      <td>06:10</td>
      <td>22:00</td>
      <td>周一至周五 06:10-22:00  周六日 06:10-22:00</td>
      <td>宁工新寓方向</td>
      <td>2</td>
      <td>LINESTRING (669104.665 3546807.526, 669135.966...</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



* conversion of the bus stations


```python
bus_stations=shp2gdf(data_dic['bus_stations'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
bus_stations.plot()
```

    original data info:(32661, 7)
    dropna-how=all,result:(32661, 7)
    dropna-several rows,result:(32661, 7)
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_07.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
bus_stations
```




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
      <th>PointName</th>
      <th>PointUid</th>
      <th>Lng</th>
      <th>Lat</th>
      <th>LineName</th>
      <th>LineUid</th>
      <th>geometry</th>
      <th>mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>窑湾街</td>
      <td>642048e0cad8f2b35abbbc6d</td>
      <td>118.772351059796</td>
      <td>32.0149082691208</td>
      <td>100路(安德门-南医大二附院总站)</td>
      <td>079645c8210ef240754ff11b</td>
      <td>POINT (667392.718 3543460.971)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>钓鱼台</td>
      <td>055169713449cef615a190d8</td>
      <td>118.774369200524</td>
      <td>32.0186971230716</td>
      <td>100路(安德门-南医大二附院总站)</td>
      <td>079645c8210ef240754ff11b</td>
      <td>POINT (667576.452 3543884.156)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>中山南路·新桥</td>
      <td>9c673bc5d281e9233617dc7a</td>
      <td>118.775609039414</td>
      <td>32.0224020014642</td>
      <td>100路(安德门-南医大二附院总站)</td>
      <td>079645c8210ef240754ff11b</td>
      <td>POINT (667686.811 3544296.826)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6</th>
      <td>中山南路·升州路</td>
      <td>7a86dc5fbdae24f696ca0727</td>
      <td>118.778042085513</td>
      <td>32.0288193940583</td>
      <td>100路(安德门-南医大二附院总站)</td>
      <td>079645c8210ef240754ff11b</td>
      <td>POINT (667904.906 3545012.080)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>三元巷</td>
      <td>ebffcf474e92bb666b49f3dd</td>
      <td>118.77894967263</td>
      <td>32.0341225821925</td>
      <td>100路(安德门-南医大二附院总站)</td>
      <td>079645c8210ef240754ff11b</td>
      <td>POINT (667980.939 3545601.438)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31766</th>
      <td>草场门·南艺·二师</td>
      <td>32d80d0496ba8643c51b75ba</td>
      <td>118.750585889996</td>
      <td>32.0624940216279</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (665251.249 3548703.091)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31767</th>
      <td>云南路</td>
      <td>e011f7eb7152db1cc16c14dd</td>
      <td>118.769458876476</td>
      <td>32.0611463898755</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (667035.513 3548582.748)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31768</th>
      <td>鼓楼</td>
      <td>b437397b104140498bfc065f</td>
      <td>118.778059469829</td>
      <td>32.0608307001415</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (667848.093 3548561.096)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31769</th>
      <td>鸡鸣寺</td>
      <td>10b942a0665ebb0dea56fa6d</td>
      <td>118.792291254236</td>
      <td>32.0595074261512</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (669194.197 3548436.618)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31770</th>
      <td>九华山</td>
      <td>eba05554756aa19431f71d2e</td>
      <td>118.800747730693</td>
      <td>32.0594338859387</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (669992.748 3548441.758)</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>4073 rows × 8 columns</p>
</div>



* conversion of the subway lines


```python
subway_lines=shp2gdf(data_dic['subway_lines'],epsg=nanjing_epsg,encoding='GBK')
subway_lines.plot()
```

    original data info:(20, 3)
    dropna-how=all,result:(20, 3)
    dropna-several rows,result:(20, 3)
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_08.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
subway_lines
```




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
      <th>LineName</th>
      <th>LineUid</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>地铁s9号线(高淳-翔宇路南)</td>
      <td>79743ab26a494368f8dc8594</td>
      <td>LINESTRING (678118.443 3469180.910, 679000.152...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>地铁s3号线(高家冲-南京南站)</td>
      <td>d4a8b8c00653bd44c17da208</td>
      <td>LINESTRING (642965.556 3531558.066, 644261.635...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>地铁s7号线(无想山-空港新城江宁)</td>
      <td>de67974fbfba8d7ca1d2309c</td>
      <td>LINESTRING (693615.069 3499371.258, 693565.026...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>LINESTRING (663540.062 3548394.957, 665251.249...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>地铁s9号线(翔宇路南-高淳)</td>
      <td>4180a0b979c3d51cf21937c6</td>
      <td>LINESTRING (672756.466 3514936.103, 677823.239...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>地铁4号线(仙林湖-龙江)</td>
      <td>417c6ecc1b171ff0b4265591</td>
      <td>LINESTRING (687490.769 3556308.016, 687908.132...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>地铁s7号线(空港新城江宁-无想山)</td>
      <td>ae7bba759b43c478c8fa5fb9</td>
      <td>LINESTRING (678334.613 3513139.261, 683072.450...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>地铁2号线(经天路-油坊桥)</td>
      <td>71110a0d5e1613f0b2077dec</td>
      <td>LINESTRING (685995.164 3555252.822, 684377.704...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>地铁s8号线(金牛湖-泰山新村)</td>
      <td>eb3196b7ee4a2a8402294fc2</td>
      <td>LINESTRING (684102.030 3593920.485, 681281.579...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>地铁3号线(秣周东路-林场)</td>
      <td>3605e86cd38d8eb386420e8a</td>
      <td>LINESTRING (672748.210 3527602.884, 672627.123...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>地铁s1号线(空港新城江宁-南京南站)</td>
      <td>6bf34b4d66d01a261811d50e</td>
      <td>LINESTRING (678334.613 3513139.261, 676986.772...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>地铁3号线(林场-秣周东路)</td>
      <td>ddf66c9323eefca1b6f42fb4</td>
      <td>LINESTRING (657111.198 3560027.940, 659506.711...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>地铁s8号线(泰山新村-金牛湖)</td>
      <td>b515b075e82e134b6d20b997</td>
      <td>LINESTRING (661349.544 3557985.428, 661648.728...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>地铁1号线(迈皋桥-中国药科大学)</td>
      <td>6ea554ece30ad8582bb97cec</td>
      <td>LINESTRING (670260.715 3553475.077, 669555.826...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>地铁10号线(雨山路-安德门)</td>
      <td>69637c486e97b11b46572623</td>
      <td>LINESTRING (652028.031 3546754.635, 653094.724...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>地铁s1号线(南京南站-空港新城江宁)</td>
      <td>9834fbeae7f43bcccb213d12</td>
      <td>LINESTRING (669278.657 3538834.591, 668192.203...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>地铁2号线(油坊桥-经天路)</td>
      <td>28f0b71195be8a4e2cb17eec</td>
      <td>LINESTRING (662164.481 3538224.996, 662258.519...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>地铁1号线(中国药科大学-迈皋桥)</td>
      <td>94ec7cb90c0014f8b6147fec</td>
      <td>LINESTRING (680533.825 3530933.483, 679610.658...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>地铁10号线(安德门-雨山路)</td>
      <td>031d4680f2fa3654e60738d4</td>
      <td>LINESTRING (665971.810 3540989.467, 664311.516...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>地铁s3号线(南京南站-高家冲)</td>
      <td>b2f388227376647a1e580045</td>
      <td>LINESTRING (669276.308 3538840.988, 667596.690...</td>
    </tr>
  </tbody>
</table>
</div>



* conversion of the subway stations


```python
subway_stations=shp2gdf(data_dic['subway_stations'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
subway_stations.plot()
```

    original data info:(348, 8)
    dropna-how=all,result:(348, 8)
    dropna-several rows,result:(348, 8)
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_09.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
subway_stations[:10]
```




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
      <th>PointName</th>
      <th>PointUid</th>
      <th>Lng</th>
      <th>Lat</th>
      <th>IsPractica</th>
      <th>LineName</th>
      <th>LineUid</th>
      <th>geometry</th>
      <th>mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34</th>
      <td>龙江</td>
      <td>cbfedaeceea4121b276e9834</td>
      <td>118.732413218273</td>
      <td>32.0599644534545</td>
      <td>1</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (663540.062 3548394.957)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>35</th>
      <td>草场门</td>
      <td>32d80d0496ba8643c51b75ba</td>
      <td>118.750585889996</td>
      <td>32.0624940216279</td>
      <td>0</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (665251.249 3548703.091)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36</th>
      <td>云南路</td>
      <td>e011f7eb7152db1cc16c14dd</td>
      <td>118.769458876476</td>
      <td>32.0611463898755</td>
      <td>1</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (667035.513 3548582.748)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>37</th>
      <td>鼓楼</td>
      <td>b437397b104140498bfc065f</td>
      <td>118.778059469829</td>
      <td>32.0608307001415</td>
      <td>0</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (667848.093 3548561.096)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>38</th>
      <td>鸡鸣寺</td>
      <td>10b942a0665ebb0dea56fa6d</td>
      <td>118.792291254236</td>
      <td>32.0595074261512</td>
      <td>0</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (669194.197 3548436.618)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>39</th>
      <td>九华山</td>
      <td>eba05554756aa19431f71d2e</td>
      <td>118.800747730693</td>
      <td>32.0594338859387</td>
      <td>0</td>
      <td>地铁4号线(龙江-仙林湖)</td>
      <td>622db20b28589d14d82f2819</td>
      <td>POINT (669992.748 3548441.758)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>70</th>
      <td>九华山</td>
      <td>02761d8a4a088ca5ebe582db</td>
      <td>118.800747730693</td>
      <td>32.0594338859387</td>
      <td>1</td>
      <td>地铁4号线(仙林湖-龙江)</td>
      <td>417c6ecc1b171ff0b4265591</td>
      <td>POINT (669992.748 3548441.758)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>71</th>
      <td>鸡鸣寺</td>
      <td>27b98eabe9b25d2190114c56</td>
      <td>118.792291254236</td>
      <td>32.0595074261512</td>
      <td>1</td>
      <td>地铁4号线(仙林湖-龙江)</td>
      <td>417c6ecc1b171ff0b4265591</td>
      <td>POINT (669194.197 3548436.618)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>72</th>
      <td>鼓楼</td>
      <td>555eae2f089a64670c0e6b40</td>
      <td>118.778059469829</td>
      <td>32.0608307001415</td>
      <td>1</td>
      <td>地铁4号线(仙林湖-龙江)</td>
      <td>417c6ecc1b171ff0b4265591</td>
      <td>POINT (667848.093 3548561.096)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>73</th>
      <td>云南路</td>
      <td>9f5b911d5ff3de4ce6e96542</td>
      <td>118.769458876476</td>
      <td>32.0611463898755</td>
      <td>0</td>
      <td>地铁4号线(仙林湖-龙江)</td>
      <td>417c6ecc1b171ff0b4265591</td>
      <td>POINT (667035.513 3548582.748)</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



* conversion of the population


```python
population=shp2gdf(data_dic['population'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
population.plot(column='Population',cmap='hot')
```

    original data info:(6673, 4)
    dropna-how=all,result:(6673, 4)
    dropna-several rows,result:(6673, 4)
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_10.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
population
```




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
      <th>Lat</th>
      <th>Lng</th>
      <th>Population</th>
      <th>geometry</th>
      <th>mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>961</th>
      <td>32.0221268324793</td>
      <td>118.715295694938</td>
      <td>386</td>
      <td>POINT (661990.598 3544174.270)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>967</th>
      <td>32.0183185999766</td>
      <td>118.724278847779</td>
      <td>2962</td>
      <td>POINT (662845.808 3543765.578)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>968</th>
      <td>32.0335505801735</td>
      <td>118.724278847779</td>
      <td>129</td>
      <td>POINT (662818.850 3545454.272)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>971</th>
      <td>32.0145102091885</td>
      <td>118.724278847779</td>
      <td>1288</td>
      <td>POINT (662852.547 3543343.361)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>974</th>
      <td>32.0259349066866</td>
      <td>118.719787271358</td>
      <td>644</td>
      <td>POINT (662408.112 3544603.197)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2605</th>
      <td>32.0906479390861</td>
      <td>118.769194611985</td>
      <td>1417</td>
      <td>POINT (666956.913 3551853.087)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2607</th>
      <td>32.0411656203557</td>
      <td>118.764703035564</td>
      <td>8886</td>
      <td>POINT (666622.721 3546360.197)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3262</th>
      <td>32.056393800478</td>
      <td>118.80512722335</td>
      <td>4121</td>
      <td>POINT (670411.876 3548111.619)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3295</th>
      <td>32.0525869930033</td>
      <td>118.80512722335</td>
      <td>4121</td>
      <td>POINT (670418.935 3547689.568)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3311</th>
      <td>32.0487800271512</td>
      <td>118.80512722335</td>
      <td>16871</td>
      <td>POINT (670425.994 3547267.499)</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>408 rows × 5 columns</p>
</div>



### 1.5 convert .csv data to GeoDataFrame
#### 1.5.1 type-A


```python
def csv2gdf_A_taxi(data_root,epsg=None,boundary=None,): #
    import glob
    from pathlib import Path
    import geopandas as gpd
    import pandas as pd
    import datetime
    # from functools import reduce
    from tqdm import tqdm
    
    suffix='csv'
    fns=glob.glob(data_root+"/*.{}".format(suffix))
    fns_stem_df=pd.DataFrame([Path(fn).stem.split('_')[:2]+[fn] for fn in fns],columns=['info','date','file_path']).set_index(['info','date'])
    g_df_dict={}
    # i=0
    for info,g in tqdm(fns_stem_df.groupby(level=0)):
        g_df=pd.concat([pd.read_csv(fn).assign(date=idx[1]) for fn in g.file_path for idx in g.index]).rename({'value':'value_{}'.format(g.index[0][0])},axis=1) 
        g_df['time']=g_df.apply(lambda row:datetime.datetime.strptime(row.date+' {}:0:0'.format(row.hour), '%Y.%m.%d %H:%S:%f'),axis=1)
        g_gdf=gpd.GeoDataFrame(g_df,geometry=gpd.points_from_xy(g_df.longitude,g_df.latitude,),crs='epsg:4326')
        # print(g_gdf)
        if epsg is not None:
            g_gdf_proj=g_gdf.to_crs(epsg=epsg)
        if boundary:
            g_gdf_proj['mask']=g_gdf_proj.geometry.apply(lambda row:row.within(boundary))
            g_gdf_proj.query('mask',inplace=True)    
        
        g_df_dict['value_{}'.format(g.index[0][0])]=g_gdf_proj
        
        # if i==1:
        #     break
        # i+=1    
    return g_df_dict
```

* conversion of the taxi data


```python
g_df_dict=csv2gdf_A_taxi(data_dic['taxi'],epsg=nanjing_epsg,boundary=boudnary_polygon)
taxi_keys=list(g_df_dict.keys())
print(taxi_keys)
g_df_dict[taxi_keys[0]].plot(column=taxi_keys[0],cmap='hot')
```

    100%|██████████| 5/5 [02:27<00:00, 29.60s/it]
    

    ['value_demand', 'value_distribute', 'value_money', 'value_response', 'value_satisfy']
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_11.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
g_df_dict[taxi_keys[0]].plot(column=taxi_keys[0],cmap='hot',markersize=3)
```




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_12.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
g_df_dict
```




    {'value_demand':       Unnamed: 0  hour  longitude  latitude  value_demand        date  \
     2              2     0   118.7601   32.0423            21  2016.04.06   
     4              4     0   118.7634   32.0889             8  2016.04.06   
     15            15     0   118.7767   32.0314            10  2016.04.06   
     23            23     0   118.7766   32.0769             5  2016.04.06   
     29            29     0   118.7965   32.0337             8  2016.04.06   
     ...          ...   ...        ...       ...           ...         ...   
     2929        2929    23   118.8012   32.0460            14  2016.04.12   
     2933        2933    23   118.7901   32.0743            10  2016.04.12   
     2939        2939    23   118.7890   32.0215             4  2016.04.12   
     2943        2943    23   118.7605   32.0777            10  2016.04.12   
     2948        2948    23   118.7639   32.0898             5  2016.04.12   
     
                         time                        geometry  mask  
     2    2016-04-06 00:00:00  POINT (666185.993 3546478.865)  True  
     4    2016-04-06 00:00:00  POINT (666413.176 3551650.338)  True  
     15   2016-04-06 00:00:00  POINT (667773.445 3545296.097)  True  
     23   2016-04-06 00:00:00  POINT (667680.961 3550340.383)  True  
     29   2016-04-06 00:00:00  POINT (669639.182 3545582.028)  True  
     ...                  ...                             ...   ...  
     2929 2016-04-12 23:00:00  POINT (670060.306 3546953.089)  True  
     2933 2016-04-12 23:00:00  POINT (668960.098 3550073.199)  True  
     2939 2016-04-12 23:00:00  POINT (668953.278 3544217.694)  True  
     2943 2016-04-12 23:00:00  POINT (666159.735 3550404.156)  True  
     2948 2016-04-12 23:00:00  POINT (666458.735 3551750.891)  True  
     
     [30212 rows x 9 columns],
     'value_distribute':        Unnamed: 0  hour  longitude  latitude  value_distribute        date  \
     0               0     0   118.7865   32.0529               312  2016.04.06   
     1               1     0   118.7944   32.0397               283  2016.04.06   
     6               6     0   118.7915   32.0319               250  2016.04.06   
     9               9     0   118.7841   32.0380               239  2016.04.06   
     10             10     0   118.7638   32.0214               234  2016.04.06   
     ...           ...   ...        ...       ...               ...         ...   
     11928       11928    23   118.7260   32.0214                42  2016.04.12   
     11930       11930    23   118.7381   32.0556                42  2016.04.12   
     11952       11952    23   118.7331   32.0306                40  2016.04.12   
     11987       11987    23   118.7348   32.0462                36  2016.04.12   
     11992       11992    23   118.7327   32.0513                36  2016.04.12   
     
                          time                        geometry  mask  
     0     2016-04-06 00:00:00  POINT (668659.545 3547695.005)  True  
     1     2016-04-06 00:00:00  POINT (669429.800 3546243.932)  True  
     6     2016-04-06 00:00:00  POINT (669170.287 3545374.622)  True  
     9     2016-04-06 00:00:00  POINT (668460.235 3546039.339)  True  
     10    2016-04-06 00:00:00  POINT (666573.235 3544167.465)  True  
     ...                   ...                             ...   ...  
     11928 2016-04-12 23:00:00  POINT (663002.922 3544109.794)  True  
     11930 2016-04-12 23:00:00  POINT (664084.761 3547919.722)  True  
     11952 2016-04-12 23:00:00  POINT (663657.168 3545140.491)  True  
     11987 2016-04-12 23:00:00  POINT (663789.933 3546872.571)  True  
     11992 2016-04-12 23:00:00  POINT (663582.561 3547434.801)  True  
     
     [93814 rows x 9 columns],
     'value_money':       Unnamed: 0  hour  longitude  latitude  value_money        date  \
     3              3     0   118.7864   32.0563        205.0  2016.04.06   
     4              4     0   118.7804   32.0680         88.0  2016.04.06   
     7              7     0   118.7279   32.0287         23.0  2016.04.06   
     8              8     0   118.7767   32.0314         12.0  2016.04.06   
     10            10     0   118.7708   32.0148         13.0  2016.04.06   
     ...          ...   ...        ...       ...          ...         ...   
     1989        1989    23   118.7605   32.0777         47.0  2016.04.12   
     1991        1991    23   118.7934   32.0635         11.0  2016.04.12   
     1997        1997    23   118.7765   32.0768         51.0  2016.04.12   
     1998        1998    23   118.7457   32.0360         24.0  2016.04.12   
     2000        2000    23   118.7639   32.0898         23.0  2016.04.12   
     
                         time                        geometry  mask  
     3    2016-04-06 00:00:00  POINT (668643.863 3548071.797)  True  
     4    2016-04-06 00:00:00  POINT (668055.953 3549359.580)  True  
     7    2016-04-06 00:00:00  POINT (663169.435 3544921.978)  True  
     8    2016-04-06 00:00:00  POINT (667773.445 3545296.097)  True  
     10   2016-04-06 00:00:00  POINT (667246.401 3543446.566)  True  
     ...                  ...                             ...   ...  
     1989 2016-04-12 23:00:00  POINT (666159.735 3550404.156)  True  
     1991 2016-04-12 23:00:00  POINT (669291.521 3548881.004)  True  
     1997 2016-04-12 23:00:00  POINT (667671.704 3550329.141)  True  
     1998 2016-04-12 23:00:00  POINT (664837.474 3545758.332)  True  
     2000 2016-04-12 23:00:00  POINT (666458.735 3551750.891)  True  
     
     [24255 rows x 9 columns],
     'value_response':       Unnamed: 0  hour  longitude  latitude  value_response        date  \
     0              0     0   118.7279   32.0287            29.0  2016.04.06   
     1              1     0   118.7965   32.0337            13.5  2016.04.06   
     9              9     0   118.7628   32.0210            41.2  2016.04.06   
     10            10     0   118.7448   32.0364            23.5  2016.04.06   
     11            11     0   118.7194   32.0518            42.5  2016.04.06   
     ...          ...   ...        ...       ...             ...         ...   
     2575        2575    23   118.7289   32.0403            60.5  2016.04.12   
     2580        2580    23   118.7934   32.0635            11.5  2016.04.12   
     2582        2582    23   118.7484   32.0633             9.0  2016.04.12   
     2587        2587    23   118.7765   32.0768             6.5  2016.04.12   
     2588        2588    23   118.7605   32.0777            11.0  2016.04.12   
     
                         time                        geometry  mask  
     0    2016-04-06 00:00:00  POINT (663169.435 3544921.978)  True  
     1    2016-04-06 00:00:00  POINT (669639.182 3545582.028)  True  
     9    2016-04-06 00:00:00  POINT (666479.506 3544121.577)  True  
     10   2016-04-06 00:00:00  POINT (664751.763 3545801.305)  True  
     11   2016-04-06 00:00:00  POINT (662325.871 3547470.151)  True  
     ...                  ...                             ...   ...  
     2575 2016-04-12 23:00:00  POINT (663243.290 3546209.527)  True  
     2580 2016-04-12 23:00:00  POINT (669291.521 3548881.004)  True  
     2582 2016-04-12 23:00:00  POINT (665043.430 3548789.101)  True  
     2587 2016-04-12 23:00:00  POINT (667671.704 3550329.141)  True  
     2588 2016-04-12 23:00:00  POINT (666159.735 3550404.156)  True  
     
     [28168 rows x 9 columns],
     'value_satisfy':        Unnamed: 0  hour  longitude  latitude  value_satisfy        date  \
     15             15     0   118.7642   32.0822            1.0  2016.04.06   
     19             19     0   118.7731   32.0723            8.0  2016.04.06   
     26             26     0   118.7965   32.0611            7.0  2016.04.06   
     27             27     0   118.7622   32.0406            7.0  2016.04.06   
     31             31     0   118.7813   32.0546            4.0  2016.04.06   
     ...           ...   ...        ...       ...            ...         ...   
     11958       11958    23   118.7512   32.0203            7.0  2016.04.12   
     11968       11968    23   118.7551   32.0613            7.0  2016.04.12   
     11975       11975    23   118.7367   32.0207            7.0  2016.04.12   
     11986       11986    23   118.7927   32.0647            6.0  2016.04.12   
     11994       11994    23   118.7368   32.0146            7.0  2016.04.12   
     
                          time                        geometry  mask  
     15    2016-04-06 00:00:00  POINT (666500.836 3550908.764)  True  
     19    2016-04-06 00:00:00  POINT (667358.957 3549824.958)  True  
     26    2016-04-06 00:00:00  POINT (669588.624 3548619.791)  True  
     27    2016-04-06 00:00:00  POINT (666387.376 3546293.628)  True  
     31    2016-04-06 00:00:00  POINT (668165.444 3547875.363)  True  
     ...                   ...                             ...   ...  
     11958 2016-04-12 23:00:00  POINT (665385.104 3544026.150)  True  
     11968 2016-04-12 23:00:00  POINT (665679.582 3548577.637)  True  
     11975 2016-04-12 23:00:00  POINT (664014.811 3544048.386)  True  
     11986 2016-04-12 23:00:00  POINT (669223.223 3549012.947)  True  
     11994 2016-04-12 23:00:00  POINT (664035.128 3543372.260)  True  
     
     [77049 rows x 9 columns]}



#### 1.5.2 type-B


```python
def csv2gdf_A_POI(fn,epsg=None,boundary=None,encoding='utf-8'): #
    import glob
    from pathlib import Path
    import geopandas as gpd
    import pandas as pd
    from tqdm import tqdm
    
    csv_df=pd.read_csv(fn,encoding=encoding)
    csv_df['superclass']=csv_df['POI类型'].apply(lambda row:row.split(';')[0])
    # print(csv_df)
    # print(csv_df.columns)
    csv_gdf=gpd.GeoDataFrame(csv_df,geometry=gpd.points_from_xy(csv_df['经度'],csv_df['纬度']),crs='epsg:4326')

    if epsg is not None:
        csv_gdf_proj=csv_gdf.to_crs(epsg=epsg)
    if boundary:
        csv_gdf_proj['mask']=csv_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        csv_gdf_proj.query('mask',inplace=True)  
    
    return csv_gdf_proj
```

* conversion of the POI


```python
POI=csv2gdf_A_POI(data_dic['POI'],epsg=nanjing_epsg,boundary=boudnary_polygon,encoding='GBK')
POI.plot(column='superclass',cmap='terrain',markersize=1)
```




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_13.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
POI
```




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
      <th>唯一ID</th>
      <th>POI名称</th>
      <th>POI类型</th>
      <th>POI类型编号</th>
      <th>行业类型</th>
      <th>地址</th>
      <th>经度</th>
      <th>纬度</th>
      <th>POI所在省份名称</th>
      <th>POI所在城市名称</th>
      <th>区域编码</th>
      <th>superclass</th>
      <th>geometry</th>
      <th>mask</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1886</th>
      <td>B0FFG6SDLE</td>
      <td>海公公平价海鲜大排档(应天大街店)</td>
      <td>餐饮服务;中餐厅;海鲜酒楼</td>
      <td>050119</td>
      <td>diner</td>
      <td>应天大街845号</td>
      <td>118.740695</td>
      <td>32.020896</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>餐饮服务</td>
      <td>POINT (664391.758 3544076.150)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1887</th>
      <td>B00190BS4Z</td>
      <td>绿杨春生态酒店(庐山路店)</td>
      <td>餐饮服务;中餐厅;综合酒楼</td>
      <td>050101</td>
      <td>diner</td>
      <td>庐山路18号</td>
      <td>118.738465</td>
      <td>32.020370</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>餐饮服务</td>
      <td>POINT (664182.129 3544014.484)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1888</th>
      <td>B00190ZBVB</td>
      <td>农家小院(黄山路店)</td>
      <td>餐饮服务;中餐厅;中餐厅</td>
      <td>050100</td>
      <td>diner</td>
      <td>黄山路天都芳庭南</td>
      <td>118.743649</td>
      <td>32.017737</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>餐饮服务</td>
      <td>POINT (664676.500 3543730.421)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1889</th>
      <td>B00190AK82</td>
      <td>麦当劳(南京应天西路店)</td>
      <td>餐饮服务;快餐厅;麦当劳</td>
      <td>050302</td>
      <td>diner</td>
      <td>南湖路58号华润苏果1层</td>
      <td>118.744616</td>
      <td>32.020215</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>餐饮服务</td>
      <td>POINT (664763.373 3544006.678)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>B00190Z6XL</td>
      <td>正大海鲜馆(应天大街店)</td>
      <td>餐饮服务;中餐厅;海鲜酒楼</td>
      <td>050119</td>
      <td>diner</td>
      <td>应天大街835-8号</td>
      <td>118.742253</td>
      <td>32.019919</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>餐饮服务</td>
      <td>POINT (664540.735 3543970.270)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>531418</th>
      <td>B00190CS40</td>
      <td>翠杉园(东南门)</td>
      <td>通行设施;临街院门;临街院门</td>
      <td>991400</td>
      <td>NaN</td>
      <td>月安街39-8号</td>
      <td>118.722027</td>
      <td>32.021447</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>通行设施</td>
      <td>POINT (662627.607 3544109.063)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>531420</th>
      <td>B0FFJ02F65</td>
      <td>都市菜园(东北门)</td>
      <td>通行设施;临街院门;临街院门</td>
      <td>991400</td>
      <td>NaN</td>
      <td>兴隆大街209号附近</td>
      <td>118.715920</td>
      <td>32.021854</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>通行设施</td>
      <td>POINT (662050.063 3544144.997)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>531421</th>
      <td>B0FFGYQBLN</td>
      <td>鸿康艺术馆(西南门)</td>
      <td>通行设施;建筑物门;建筑物正门</td>
      <td>991001</td>
      <td>NaN</td>
      <td>松花江西街28-2号</td>
      <td>118.719849</td>
      <td>32.018001</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>通行设施</td>
      <td>POINT (662427.939 3543723.651)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>531424</th>
      <td>B0FFG4F2XG</td>
      <td>养墨堂美术馆(东北门)</td>
      <td>通行设施;临街院门;临街院正门</td>
      <td>991401</td>
      <td>NaN</td>
      <td>月安街39-8号附近</td>
      <td>118.721976</td>
      <td>32.021409</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>通行设施</td>
      <td>POINT (662622.862 3544104.778)</td>
      <td>True</td>
    </tr>
    <tr>
      <th>531425</th>
      <td>B0FFF2DGVM</td>
      <td>好邻居生活广场(东北门)</td>
      <td>通行设施;建筑物门;建筑物正门</td>
      <td>991001</td>
      <td>NaN</td>
      <td>兴隆大街209号</td>
      <td>118.715882</td>
      <td>32.021740</td>
      <td>江苏省</td>
      <td>南京市</td>
      <td>建邺区</td>
      <td>通行设施</td>
      <td>POINT (662046.680 3544132.303)</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>126149 rows × 14 columns</p>
</div>



### 1.6 convert .db(SQLite) data to GeoDataFrame


```python
def db2df(database_sql,table):
    import sqlite3
    import pandas as pd
    
    conn=sqlite3.connect(database_sql)
    df=pd.read_sql_query("SELECT * from {}".format(table), conn)
    # print(df)
    
    return df
```

* conversion of the Metro Weibo(microblog) publish


```python
microblog=db2df(data_dic['microblog'],'NajingMetro')
microblog
```




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
      <th>DATE</th>
      <th>LineALL</th>
      <th>Line1</th>
      <th>Line2</th>
      <th>Line3</th>
      <th>Line4</th>
      <th>Line10</th>
      <th>LineS1</th>
      <th>LineS3</th>
      <th>LineS7</th>
      <th>LineS8</th>
      <th>LineS9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-10-27</td>
      <td>355.5</td>
      <td>102.2</td>
      <td>92.3</td>
      <td>85.9</td>
      <td>22.2</td>
      <td>18.5</td>
      <td>10.3</td>
      <td>8.1</td>
      <td>1.4</td>
      <td>12.1</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-10-26</td>
      <td>382.7</td>
      <td>112.1</td>
      <td>96.3</td>
      <td>91.6</td>
      <td>25.2</td>
      <td>22.5</td>
      <td>10.3</td>
      <td>8.7</td>
      <td>1.4</td>
      <td>12.6</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-10-24</td>
      <td>332.4</td>
      <td>97.5</td>
      <td>84.7</td>
      <td>80.7</td>
      <td>22.5</td>
      <td>18.2</td>
      <td>8.0</td>
      <td>7.5</td>
      <td>1.0</td>
      <td>10.8</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-10-23</td>
      <td>330.2</td>
      <td>96.4</td>
      <td>84.2</td>
      <td>80.1</td>
      <td>22.4</td>
      <td>18.2</td>
      <td>7.9</td>
      <td>7.6</td>
      <td>1.1</td>
      <td>10.8</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-10-22</td>
      <td>326.2</td>
      <td>95.7</td>
      <td>81.7</td>
      <td>79.6</td>
      <td>22.0</td>
      <td>18.2</td>
      <td>8.1</td>
      <td>7.7</td>
      <td>1.0</td>
      <td>10.8</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>962</th>
      <td>2016-02-06</td>
      <td>109.2</td>
      <td>38.5</td>
      <td>30.4</td>
      <td>26.2</td>
      <td>0.0</td>
      <td>5.2</td>
      <td>3.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>963</th>
      <td>2016-02-05</td>
      <td>154.9</td>
      <td>55.7</td>
      <td>44.8</td>
      <td>35.9</td>
      <td>0.0</td>
      <td>7.7</td>
      <td>3.9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.9</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>964</th>
      <td>2016-02-04</td>
      <td>182.2</td>
      <td>65.9</td>
      <td>54.0</td>
      <td>41.5</td>
      <td>0.0</td>
      <td>9.4</td>
      <td>4.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.3</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>965</th>
      <td>2016-02-03</td>
      <td>199.6</td>
      <td>71.7</td>
      <td>59.8</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>10.4</td>
      <td>4.2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.5</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>966</th>
      <td>2016-02-02</td>
      <td>208.6</td>
      <td>74.7</td>
      <td>63.2</td>
      <td>47.7</td>
      <td>0.0</td>
      <td>10.9</td>
      <td>4.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.7</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>967 rows × 12 columns</p>
</div>



### 1.7 convert .xls(excel) data to GeoDataFrame


```python
def xls2gdf(fn,epsg=None,boundary=None,sheet_name=0):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import LineString
    
    xls_df=pd.read_excel(fn,sheet_name=sheet_name)
    # print(xls_df)
    # print(xls_df.columns)
    xls_df['route_line']=xls_df.apply(lambda row:LineString([(row['开始维度'],row['开始经度'],),(row['结束维度'],row['结束经度'],)]),axis=1)
    xls_gdf=gpd.GeoDataFrame(xls_df,geometry=xls_df.route_line,crs='epsg:4326')    
    
    # print(xls_df)
    if epsg is not None:
        xls_gdf_proj=xls_gdf.to_crs(epsg=epsg)
    if boundary:
        xls_gdf_proj['mask']=xls_gdf_proj.geometry.apply(lambda row:row.within(boundary))
        xls_gdf_proj.query('mask',inplace=True)     
    
    return xls_gdf_proj
```

* conversion of the bick sharing(no data were available for Nanjing area)


```python
bike_sharing=xls2gdf(data_dic['bike_sharing'],epsg=nanjing_epsg,sheet_name='共享单车数据a')#boundary=boudnary_polygon,
bike_sharing.plot()
```




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_14.png" height='auto' width='auto' title="caDesign"></a> 
    


### 1.8 Sentinel-2 remote sensing image processing

* read the information and calculate NDVI


```python
def Sentinel2_bandFNs(MTD_MSIL2A_fn):
    import xml.etree.ElementTree as ET
    '''
    funciton - 获取sentinel-2波段文件路径，和打印主要信息
    
    Paras:
    MTD_MSIL2A_fn - MTD_MSIL2A 文件路径
    
    Returns:
    band_fns_list - 波段相对路径列表
    band_fns_dict - 波段路径为值，反应波段信息的字段为键的字典
    '''
    Sentinel2_tree=ET.parse(MTD_MSIL2A_fn)
    Sentinel2_root=Sentinel2_tree.getroot()

    print("GENERATION_TIME:{}\nPRODUCT_TYPE:{}\nPROCESSING_LEVEL:{}".format(Sentinel2_root[0][0].find('GENERATION_TIME').text,
                                                           Sentinel2_root[0][0].find('PRODUCT_TYPE').text,                 
                                                           Sentinel2_root[0][0].find('PROCESSING_LEVEL').text
                                                          ))
    
    # print("MTD_MSIL2A.xml 文件父结构:")
    for child in Sentinel2_root:
        print(child.tag,"-",child.attrib)
    print("_"*50)    
    band_fns_list=[elem.text for elem in Sentinel2_root.iter('IMAGE_FILE')] #[elem.text for elem in Sentinel2_root[0][0][11][0][0].iter()]
    band_fns_dict={f.split('_')[-2]+'_'+f.split('_')[-1]:f+'.jp2' for f in band_fns_list}
    # print('get sentinel-2 bands path:\n',band_fns_dict)
    
    return band_fns_list,band_fns_dict  
```


```python
# Function to normalize the grid values
def normalize_(array):
    """
    function - 数组标准化 Normalizes numpy arrays into scale 0.0 - 1.0
    """
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def sentinel_2_NDVI(sentinel_2_root,save_path):
    import os
    import earthpy.spatial as es
    import rasterio as rio
    from tqdm import tqdm
    import shapely
    import numpy as np
    from scipy import stats
    from osgeo import gdal
    
    MTD_fn=os.path.join(sentinel_2_root,'MTD_MSIL2A.xml')
    band_fns_list,band_fns_dict=Sentinel2_bandFNs(MTD_fn) 
    # print(band_fns_dict).
    bands_selection=["B02_10m","B03_10m","B04_10m","B08_10m"] 
    stack_bands=[os.path.join(sentinel_2_root,band_fns_dict[b]) for b in bands_selection]
    # print(stack_bands)
    array_stack, meta_data=es.stack(stack_bands)    
    meta_data.update(
        count=1,
        dtype=rio.float64,
        driver='GTiff'
        )   
    print("meta_data:\n",meta_data)   

    NDVI=(array_stack[3]-array_stack[2])/(array_stack[3]+array_stack[2])    
    with rio.open(save_path,'w',**meta_data) as dst:
        dst.write(np.expand_dims(NDVI.astype(meta_data['dtype']),axis=0))
    print('NDVI has been saved as raster .tif format....')
    
    return NDVI
```


```python
ndvi_fn=r'C:\Users\richi\omen_richiebao\omen_IIIT\workshop_LA_UP_iit\data\RS\NDVI.tif'
sentinel_2_NDVI=sentinel_2_NDVI(data_dic['sentinel_2'],ndvi_fn)
```

    GENERATION_TIME:2020-08-19T04:51:47.000000Z
    PRODUCT_TYPE:S2MSI2A
    PROCESSING_LEVEL:Level-2A
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}General_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Geometric_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Auxiliary_Data_Info - {}
    {https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd}Quality_Indicators_Info - {}
    __________________________________________________
    meta_data:
     {'driver': 'GTiff', 'dtype': 'float64', 'nodata': None, 'width': 10980, 'height': 10980, 'count': 1, 'crs': CRS.from_epsg(32650), 'transform': Affine(10.0, 0.0, 600000.0,
           0.0, -10.0, 3600000.0), 'blockxsize': 1024, 'blockysize': 1024, 'tiled': True}
    

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\ipykernel_launcher.py:33: RuntimeWarning: invalid value encountered in true_divide
    

    NDVI has been saved as raster .tif format....
    


```python
sentinel_2_NDVI
```




    array([[0.62184142, 0.56410256, 0.46419873, ..., 0.70993548, 0.81875826,
            0.81105528],
           [0.50374774, 0.41753607, 0.36660413, ..., 0.77303071, 0.80636605,
            0.81873727],
           [0.329653  , 0.30541788, 0.24942792, ..., 0.78369987, 0.80998227,
            0.8226968 ],
           ...,
           [0.79490053, 0.79994167, 0.81625442, ..., 0.79172611, 0.82611859,
            0.7914024 ],
           [0.791608  , 0.79526627, 0.81345029, ..., 0.86797066, 0.86704995,
            0.85682889],
           [0.80112873, 0.80350085, 0.81279343, ..., 0.87049029, 0.86832298,
            0.84456588]])



* raster cropping


```python
def raster_crop(raster_fn,crop_shp_fn,boundary=None):
    import rasterio as rio
    import geopandas as gpd
    import earthpy.spatial as es
    import numpy as np
    import earthpy.plot as ep
    from shapely.geometry import shape
    
    with rio.open(raster_fn) as src:
        ori_raster=src.read(1)
        ori_profile=src.profile
    print(ori_raster.shape)
    crop_boundary=gpd.read_file(crop_shp_fn).to_crs(ori_profile['crs'])
    # print(crop_boundary)
    print("_"*50)
    print(' crop_boundary: {}'.format(crop_boundary.crs))
    print("_"*50)
    print(' ori_raster: {}'.format( ori_profile['crs']))
    
    with rio.open(raster_fn) as src:
        cropped_img, cropped_meta=es.crop_image(src,crop_boundary)
    print(cropped_img.shape)
    
    cropped_meta.update({"driver": "GTiff",
                         "height": cropped_img.shape[0],
                         "width":  cropped_img.shape[1],
                         "transform": cropped_meta["transform"]})
    cropped_img_mask=np.ma.masked_equal(cropped_img[0], -9999.0) 
    # print(cropped_img_mask)
    ep.plot_bands(cropped_img_mask, cmap='terrain', cbar=False) 
    print(type(cropped_img_mask))
    
    cropped_shapes=(
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(rio.features.shapes(cropped_img.astype(np.float32),transform=cropped_meta['transform']))) #,mask=None
    # print(cropped_shapes)
    geoms=list(cropped_shapes)    
    print(geoms[0])
    cropped_img_gpd=gpd.GeoDataFrame.from_features(geoms)
    cropped_img_gpd.geometry=cropped_img_gpd.geometry.apply(lambda row:row.centroid)
    # print(cropped_img_gpd)

    if boundary:
        cropped_img_gpd['mask']=cropped_img_gpd.geometry.apply(lambda row:row.within(boundary))
        cropped_img_gpd.query('mask',inplace=True)      
    
    return cropped_img_gpd
```


```python
ndvi_cropped=raster_crop(raster_fn=ndvi_fn,crop_shp_fn='./data/GIS/b_centroid_buffer.shp',boundary=boudnary_polygon) #,cropped_fn='./data/GIS/NDVI_cropped.tif'
ndvi_cropped.plot(column='raster_val',cmap='terrain',markersize=1)
```

    (10980, 10980)
    __________________________________________________
     crop_boundary: PROJCS["WGS 84 / UTM zone 50N",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",117],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","32650"]]
    __________________________________________________
     ori_raster: EPSG:32650
    (1, 1001, 1001)
    


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_15.png" height='auto' width='auto' title="caDesign"></a> 
    


    <class 'numpy.ma.core.MaskedArray'>
    {'properties': {'raster_val': 0.25824177265167236}, 'geometry': {'type': 'Polygon', 'coordinates': [[(660440.0, 3552590.0), (660440.0, 3552580.0), (660450.0, 3552580.0), (660450.0, 3552590.0), (660440.0, 3552590.0)]]}}
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_16.png" height='auto' width='auto' title="caDesign"></a> 
    


### 1.9 Writes GeoDataFrame format data to the database

* write GeoDataFrame into SQLite database


```python
class SQLite_handle():
    def __init__(self,db_file):
        self.db_file=db_file   
    
    def create_connection(self):
        import sqlite3
        from sqlite3 import Error
        """ create a database connection to a SQLite database """
        conn=None
        try:
            conn=sqlite3.connect(self.db_file)
            print('connected.',"SQLite version:%s"%sqlite3.version,)
        except Error as e:
            print(e)
        finally:
            if conn:
                conn.close()
```

create or connect to the SQLite database


```python
db_file=r'./database/workshop_LAUP_iit.db'
sql_w=SQLite_handle(db_file)
sql_w.create_connection()  
```

    connected. SQLite version:2.6.0
    


```python
def gpd2SQLite(gdf_,db_fp,table_name):
    from geoalchemy2 import Geometry, WKTElement
    from sqlalchemy import create_engine
    import pandas as pd    
    import copy
    import shapely.wkb    

    gdf=copy.deepcopy(gdf_)
    crs=gdf.crs
    # print(help(crs))
    # print(crs.to_epsg())
    # gdf['geom']=gdf['geometry'].apply(lambda g: WKTElement(g.wkt,srid=crs.to_epsg()))
    #convert all values from the geopandas geometry column into their well-known-binary representations
    gdf['geom']=gdf.apply(lambda row: shapely.wkb.dumps(row.geometry),axis=1)
    gdf.drop(columns=['geometry','mask'],inplace=True)
    # print(type(gdf.geom.iloc[0]))
    print(gdf)
    engine=create_engine('sqlite:///'+'\\\\'.join(db_fp.split('\\')),echo=True)
    gdf.to_sql(table_name, con=engine, if_exists='replace', index=False,) #dtype={'geometry': Geometry('POINT')} ;dtype={'geometry': Geometry('POINT',srid=crs.to_epsg())}
    print('has been written to into the SQLite database...')
```


```python
gpd2SQLite(population,db_file,table_name='population')
```

                       Lat               Lng Population  \
    961   32.0221268324793  118.715295694938        386   
    967   32.0183185999766  118.724278847779       2962   
    968   32.0335505801735  118.724278847779        129   
    971   32.0145102091885  118.724278847779       1288   
    974   32.0259349066866  118.719787271358        644   
    ...                ...               ...        ...   
    2605  32.0906479390861  118.769194611985       1417   
    2607  32.0411656203557  118.764703035564       8886   
    3262   32.056393800478   118.80512722335       4121   
    3295  32.0525869930033   118.80512722335       4121   
    3311  32.0487800271512   118.80512722335      16871   
    
                                                       geom  
    961   b'\x01\x01\x00\x00\x00\xca|\x1c2\xcd3$A\xea\x0...  
    967   b'\x01\x01\x00\x00\x00\xdb\xc8\xda\x9d{:$AN\xf...  
    968   b'\x01\x01\x00\x00\x00\xc3a4\xb3E:$A\xce\xff\x...  
    971   b'\x01\x01\x00\x00\x00\xa1I\xf4\x17\x89:$A\xa6...  
    974   b"\x01\x01\x00\x00\x00.\xbc'9\x107$A5\xa5(\x99...  
    ...                                                 ...  
    2605  b'\x01\x01\x00\x00\x000~\x7f\xd3\x99Z$A\xcb\x1...  
    2607  b'\x01\x01\x00\x00\x00\x14\xcf6q\xfdW$Ab(-\x19...  
    3262  b'\x01\x01\x00\x00\x00\xe4\xe9j\xc0\x97u$AB\x8...  
    3295  b'\x01\x01\x00\x00\x00\t\xd8\xd0\xde\xa5u$A\xe...  
    3311  b'\x01\x01\x00\x00\x00\x9e\x95\xfa\xfc\xb3u$Au...  
    
    [408 rows x 4 columns]
    2021-02-28 23:29:42,834 INFO sqlalchemy.engine.base.Engine SELECT CAST('test plain returns' AS VARCHAR(60)) AS anon_1
    2021-02-28 23:29:42,835 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,836 INFO sqlalchemy.engine.base.Engine SELECT CAST('test unicode returns' AS VARCHAR(60)) AS anon_1
    2021-02-28 23:29:42,836 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,837 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("population")
    2021-02-28 23:29:42,837 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,839 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_info("population")
    2021-02-28 23:29:42,839 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,840 INFO sqlalchemy.engine.base.Engine SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
    2021-02-28 23:29:42,841 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,842 INFO sqlalchemy.engine.base.Engine PRAGMA main.table_xinfo("population")
    2021-02-28 23:29:42,843 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,844 INFO sqlalchemy.engine.base.Engine SELECT sql FROM  (SELECT * FROM sqlite_master UNION ALL   SELECT * FROM sqlite_temp_master) WHERE name = ? AND type = 'table'
    2021-02-28 23:29:42,844 INFO sqlalchemy.engine.base.Engine ('population',)
    2021-02-28 23:29:42,845 INFO sqlalchemy.engine.base.Engine PRAGMA main.foreign_key_list("population")
    2021-02-28 23:29:42,846 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,846 INFO sqlalchemy.engine.base.Engine PRAGMA temp.foreign_key_list("population")
    2021-02-28 23:29:42,847 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,847 INFO sqlalchemy.engine.base.Engine SELECT sql FROM  (SELECT * FROM sqlite_master UNION ALL   SELECT * FROM sqlite_temp_master) WHERE name = ? AND type = 'table'
    2021-02-28 23:29:42,848 INFO sqlalchemy.engine.base.Engine ('population',)
    2021-02-28 23:29:42,849 INFO sqlalchemy.engine.base.Engine PRAGMA main.index_list("population")
    2021-02-28 23:29:42,850 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,851 INFO sqlalchemy.engine.base.Engine PRAGMA temp.index_list("population")
    2021-02-28 23:29:42,852 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,852 INFO sqlalchemy.engine.base.Engine PRAGMA main.index_list("population")
    2021-02-28 23:29:42,853 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,854 INFO sqlalchemy.engine.base.Engine PRAGMA temp.index_list("population")
    2021-02-28 23:29:42,854 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,855 INFO sqlalchemy.engine.base.Engine SELECT sql FROM  (SELECT * FROM sqlite_master UNION ALL   SELECT * FROM sqlite_temp_master) WHERE name = ? AND type = 'table'
    2021-02-28 23:29:42,855 INFO sqlalchemy.engine.base.Engine ('population',)
    2021-02-28 23:29:42,858 INFO sqlalchemy.engine.base.Engine 
    DROP TABLE population
    2021-02-28 23:29:42,858 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,865 INFO sqlalchemy.engine.base.Engine COMMIT
    2021-02-28 23:29:42,867 INFO sqlalchemy.engine.base.Engine 
    CREATE TABLE population (
    	"Lat" TEXT, 
    	"Lng" TEXT, 
    	"Population" TEXT, 
    	geom TEXT
    )
    
    
    2021-02-28 23:29:42,867 INFO sqlalchemy.engine.base.Engine ()
    2021-02-28 23:29:42,872 INFO sqlalchemy.engine.base.Engine COMMIT
    2021-02-28 23:29:42,874 INFO sqlalchemy.engine.base.Engine BEGIN (implicit)
    2021-02-28 23:29:42,877 INFO sqlalchemy.engine.base.Engine INSERT INTO population ("Lat", "Lng", "Population", geom) VALUES (?, ?, ?, ?)
    2021-02-28 23:29:42,878 INFO sqlalchemy.engine.base.Engine (('32.0221268324793', '118.715295694938', '386', b'\x01\x01\x00\x00\x00\xca|\x1c2\xcd3$A\xea\x0b\x7f"7\nKA'), ('32.0183185999766', '118.724278847779', '2962', b'\x01\x01\x00\x00\x00\xdb\xc8\xda\x9d{:$AN\xf0\x03\xcaj\tKA'), ('32.0335505801735', '118.724278847779', '129', b'\x01\x01\x00\x00\x00\xc3a4\xb3E:$A\xce\xff\xdc"\xb7\x0cKA'), ('32.0145102091885', '118.724278847779', '1288', b'\x01\x01\x00\x00\x00\xa1I\xf4\x17\x89:$A\xa6\xefB\xae\x97\x08KA'), ('32.0259349066866', '118.719787271358', '644', b"\x01\x01\x00\x00\x00.\xbc'9\x107$A5\xa5(\x99\r\x0bKA"), ('32.029742822588', '118.719787271358', '258', b'\x01\x01\x00\x00\x00r\xcb`\xc7\x027$A3\xcc4\xae\xe0\x0bKA'), ('32.0335505801735', '118.719787271358', '515', b'\x01\x01\x00\x00\x00\xa8f`U\xf56$A}b\t\xc1\xb3\x0cKA'), ('32.0183185999766', '118.719787271358', '2318', b'\x01\x01\x00\x00\x00</\t\x1c+7$A6\xcbihg\tKA')  ... displaying 10 of 408 total bound parameter sets ...  ('32.0525869930033', '118.80512722335', '4121', b'\x01\x01\x00\x00\x00\t\xd8\xd0\xde\xa5u$A\xe0`\xaa\xc8\x14\x11KA'), ('32.0487800271512', '118.80512722335', '16871', b'\x01\x01\x00\x00\x00\x9e\x95\xfa\xfc\xb3u$Au^\xd5\xbfA\x10KA'))
    2021-02-28 23:29:42,881 INFO sqlalchemy.engine.base.Engine COMMIT
    has been written to into the SQLite database...
    

You can install [DB Browser for SQLite](https://sqlitebrowser.org/) to view your data loaded.

<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_01.png" height='auto' width='800' title="caDesign"></a> 

* writes GeoDataFrame into PostSQL database and reads

[PostgreSQL](https://www.postgresql.org/) needs to be installed, with postgis-bundle; and [pgAdmin](https://www.pgadmin.org/).


```python
def gpd2postSQL(gdf,table_name,**kwargs):
    from sqlalchemy import create_engine
    # engine=create_engine("postgres://postgres:123456@localhost:5432/workshop-LA-UP_IIT")  
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf.to_postgis(table_name, con=engine, if_exists='replace', index=False,)  
    print("_"*50)
    print('has been written into the PostSQL database...')
    
def postSQL2gpd(table_name,geom_col='geometry',**kwargs):
    from sqlalchemy import create_engine
    import geopandas as gpd
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostSQL database...')    
    return gdf
```


```python
gpd2postSQL(population,table_name='population',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
```

    __________________________________________________
    has been written into the PostSQL database...
    


```python
population_postsql=postSQL2gpd(table_name='population',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
population_postsql.plot(column='Population',cmap='hot')
```

    __________________________________________________
    The data has been read from PostSQL database...
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_17.png" height='auto' width='auto' title="caDesign"></a> 
    


* view the data in pgAdmin

<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_02.png" height='auto' width='1200' title="caDesign"></a> 

* QGIS links to the PostSQL dataset to open and view the data.

<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_03.png" height='auto' width='1200' title="caDesign"></a> 

Raster data is written into the PostSQL database. We need to open the windows terminal(cmd), locate to 'C: Program Files\PostgreSQL\13\bin', and use the tool 'raster2pgsql.exe' write the raster data. Not recommended, large raster takes time, and it is possible not to be open with QGIS.

The command entered at the terminal is: `raster2pgsql -s 32650 -I -M "C:\Users\richi\omen_richiebao\omen_IIIT\workshop_LA_UP_iit\data\RS\NDVI.tif" -F -t 10x10 public.NDVI | psql -d workshop-LA-UP_IIT -U postgres -h localhost -p 5432`. generated by the following code.


```python
def raster2postSQL(raster_fn,**kwargs):
    from osgeo import gdal, osr
    import psycopg2
    import subprocess
    from pathlib import Path
    
    raster=gdal.Open(raster_fn)
    # print(raster)
    proj=osr.SpatialReference(wkt=raster.GetProjection())
    print(proj)
    projection=str(proj.GetAttrValue('AUTHORITY',1))
    gt=raster.GetGeoTransform()
    pixelSizeX=str(round(gt[1]))
    pixelSizeY=str(round(-gt[5]))
    
    # cmds='raster2pgsql -s '+projection+' -I -C -M "'+raster_fn+'" -F -t '+pixelSizeX+'x'+pixelSizeY+' public.'+'uu'+' | psql -d {mydatabase} -U {myusername} -h localhost -p 5432'.format(mydatabase=kwargs['mydatabase'],myusername=kwargs['myusername'])
    cmds='raster2pgsql -s '+projection+' -I -M "'+raster_fn+'" -F -t '+pixelSizeX+'x'+pixelSizeY+' public.'+Path(raster_fn).stem+' | psql -d {mydatabase} -U {myusername} -h localhost -p 5432'.format(mydatabase=kwargs['mydatabase'],myusername=kwargs['myusername'])
    print("_"*50)
    print(cmds)
    subprocess.call(cmds, shell=True)
    print("_"*50)
    print('The raster has been loaded into PostSQL...')
```


```python
raster2postSQL(ndvi_fn,table_name='ndvi',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
```

    PROJCS["WGS 84 / UTM zone 50N",
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]],
        PROJECTION["Transverse_Mercator"],
        PARAMETER["latitude_of_origin",0],
        PARAMETER["central_meridian",117],
        PARAMETER["scale_factor",0.9996],
        PARAMETER["false_easting",500000],
        PARAMETER["false_northing",0],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AXIS["Easting",EAST],
        AXIS["Northing",NORTH],
        AUTHORITY["EPSG","32650"]]
    __________________________________________________
    raster2pgsql -s 32650 -I -M "C:\Users\richi\omen_richiebao\omen_IIIT\workshop_LA_UP_iit\data\RS\NDVI.tif" -F -t 10x10 public.NDVI | psql -d workshop-LA-UP_IIT -U postgres -h localhost -p 5432
    __________________________________________________
    The raster has been loaded into PostSQL...
    



> [workshop_LAUP_IIT_database.sql download link](https://github.com/richieBao/guide_to_digitalDesign_of_LAUPArhi_knowledgeStruc/tree/main/database)


<a href=""><img src="./workshop-LA-UP_IIT/imgs/datapreprocessing_19.png" height='auto' width='auto' title="caDesign"></a> 
