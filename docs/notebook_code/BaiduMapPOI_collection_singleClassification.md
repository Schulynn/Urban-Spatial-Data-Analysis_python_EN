> Created on Tue Sep 12 15/58/43 2017  @author: Richie Bao-caDesign(cadesign.cn)
> __+updated Tue Jun 16 14/26/28 2020 by Richie Bao

# Baidu Map POI data crawler and geospatial point map
[JupterLab .ipynb file download location](https://github.com/richieBao/Urban-Spatial-Data-Analysis_python_EN/tree/master/notebook/BaiduMapPOIcollection_ipynb)

In the space analysis, it will inevitably involve two aspects of data. One kind is the entity's physical space (emphasis on geometric meaning). It also can be divided into 2d plane data (such as remote sensing image, the city map, etc.). And three-dimensional space data (such as radar data (high altitude and ground)) scan, the general sense of various formats of urban 3d model, etc.). And all kinds of attribute data carried by physical space. One sort of attribute data reflects two-dimensional plane data (combined with two-dimensional geographical location, can be classified into two-dimensional plane data), such as land-use type, natural resource distribution, address name, etc.  And data reflects the nature of activities of humans, animals, and intangible substances, such as people's activity tracks(taxi and bike-sharing tracks, night lights, users number based on mobile phone base stations, etc.), animal migration paths and changes in various microclimate measurements.

[Baidu Map open platform](http://lbsyun.baidu.com/index.php?title=%E9%A6%96%E9%A1%B5) provides map-related functions and services, and its Web service API provides HTTP/HTTPS interface for developers, that is, developers initiate retrieval requests in the form of HTTP/HTTPS to obtain retrieved data in JSON or XML format. Among them, [POI (points of interest) data](http://lbsyun.baidu.com/index.php?title=lbscloud/poitags) currently includes 21 major categories and small categories, which are data (business distribution) reflecting the attributes of human activities carried by physical space, as follows:


| primary trade classification| secondary trade classification                                                                                                                                                                         |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| food          |   Chinese restaurant, foreign restaurant, snack bar, cake, and dessert shop, coffee shop, teahouse, bar, other                                                                                                                  |
|  hotel        |  star hotel, express hotel, apartment hotel, homestay, others                                                                                                                                          |
|  shop        |   shopping centers,  department stores, supermarkets, convenience stores, home building materials, digital appliances, shops, markets, others                                                                                                              |
|  life service    |  communications business office, post office, logistics company, ticket office, laundry, graphic fast print shop, photo studio, real estate agency, public utility, maintenance point, housekeeping service, funeral service, lottery sales point, pet service, newspaper kiosk, public toilet, bicycle road station,others  |
|  beauty        |  beauty, hair, manicure, body care, others                                                                                                                                                        |
|  tourist attractions    |  parks, zoos, botanical gardens, amusement parks, museums, aquariums, beaches, cultural relics, monuments, churches, scenic spots, temples, others                                                                                    |
|  leisure entertainment    |  holiday village, farmyard, cinema, KTV, theatre, dance hall, internet bar, game place, bath massage, leisure square, others                                                                                                 |
|  sports fitness    |  sports venues, extreme sports venues, fitness centers, others                                                                                                                                              |
| education and training    |  universities, secondary schools, primary schools, kindergartens, adult education, parent-child education, special education schools, overseas study intermediaries, scientific research institutions, training institutions, libraries, science and technology museums, others                                                             |
|  culture media    |  press and publication, radio and television, art groups, art galleries, exhibition hall, cultural palaces, others                                                                                                                          |
|  medical        |  general hospitals, specialized hospitals, clinics, pharmacies, physical examination institutions, sanitariums, first aid centers, CDC, medical devices, medical care, others                                                                                      |
|  auto service   |  car sales, car maintenance, car beauty, car parts, car rental, car testing, others                      airport, railway station, subway station, subway line, long-distance bus station, bus station, bus line, port, parking lot, refueling station, service area, toll station, bridge, charging station, roadside parking space, ordinary parking s[ace, pick-up point, others                       |
|  financial        |  banks, ATM, credit unions, investment banking, pawnshops, others                                                                                                                                           |
|  real estate      |  office buildings, residential areas, dormitories, internal buildings, others                                                                                                                                                |
| companies    |  companies, parks, agriculture, forestry, horticulture, factories and mines, others                                                                                                                                                    |
|  government agencies    |  central office, governments at all levels, administrative units, public procuratorial and judicial center, foreign-related institutions, political parties and organizations, welfare institutions, political and educational institutions, social organizations, democratic parties, neighborhood committees, others                                                           |
|  entrance and exit      |  expressway exit, expressway entrances, airport exits, airport entrances, station exit, station entrances, dorrs(note: doors of buildings and architectural complex), parking lot exits, high-speed bicycle exits, high-speed bicycle entrances, others          |
|  natural feature    |   islands, mountains, water systems, others                                                                                                                                                             |
|  landmark    |   provinces, provincial-level cities, prefecture-level cities, districts, countries, business districts, towns, villages, others                                                                                                                                  |
|  door address       |   gate address, others  

> From：[Baidu Map Open Platform](http://lbsyun.baidu.com/index.php?title=lbscloud/poitags) It was updated on May 11, 2020。

## 1. Baidu Map POI data crawler-single classification implementation with a geospatial point map
### 1.1 Single classification POI crawl
To crawl data, we need to view the retrieval method provided by Baidu and configure corresponding parameters to download according to its requirements. See the [service documentation](http://lbsyun.baidu.com/index.php?title=webapi/guide/webservice-placeapi) for details. Its experiment statement for relevant configuration retrieval, rectangle area search example: `http://api.map.baidu.com/place/v2/search?query=bank&bounds=39.915,116.404,39.975,116.414&output=json&ak={your key} //GET request`. The statement of the Baidu map example only contains several request parameters. There are about 15 retrieval parameters of the rectangle area search method, which can be used according to the data download requirements to determine which request parameters are used. At the same time, it also provides a circular area search, administrative area search, location details search services. The request parameters for rectangle area search retrieval are configured as follows:
```python
query_dic={
   'query':Retrieve the keyword. Circular region retrieval and rectangular region retrieval support multiple keyword union retrievals, different keywords are separated by \$, up to 10 keywords retrieval. For example:”银行$酒店.” If you need to retrieve by POI category, the category is set by query parameter, such as query='旅游景点'.,
   'page_size': The number of POI recalled in a single recall, with ten records by default and a maximum twenty returned. When retrieving multiple keywords, the number of records returned is the number of keywords*page_size, for example,  Page_size ='20', 
   'page_num':paging page number defaults to 0 for the first page, 1 for the second page, and so on. Often used with page_size, such as page_num='0',
   'scope':Value 1 or null returns basic information. Value 2 returns to retrieve POI details, such as scope='2',
   'bounds':The rectangular area retrieval with the coordinates in the lower left and upper right, separated by ',', for example, str(leftBottomCoordi[1]) + ',' + str(leftBottomCoordi[0]) + ','+str(rightTopCoordi[1]) + ',' + str(rightTopCoordi[0]),
   'output':Output in JSON or XML, for example, output='json',
   'ak':Developer's access key required. Before V2, this property was key.               
}
```
The request parameter for the data downloading is required to request an access password 'AK.' Note that you need to register to log in, [address](http://lbsyun.baidu.com/apiconsole/key/create). Let us start by configuring the necessary parameters. The ultimate goal is to store the download data as .csv(Comma-Separated Values, storing tabular data in plain text) and .json(JavaScript Object Notation, lightweight data-interchange format, concise and clear hierarchy) data format, respectively. Therefore, 'json' and 'csv' libraries are imported to assist in reading and writing files in corresponding formats. Because the data is to be retrieved through the web address, the HTTP library, 'urllib', is imported to implement the corresponding URL (Uniform Resource Locator, network address) processing. The 'os' and 'pathlib' libraries are available for file path management.


```python
import urllib, json, csv,os,pathlib
data_path='./data' #Configure the data storage location
#Define the store file name
poi_fn_csv=os.path.join(data_path,'poi_csv.csv')
poi_fn_json=os.path.join(data_path,'poi_json.json')
```

Configure the request parameters, noting that 'page_num' parameter is page increments and the page range `page_num_range=range(20)`for the initial parameter. The output parameter is configured directly as a fixed 'json', so it is implemented within the function. At the same time, due to the limitation of Baidu API, the amount of POI data returned within the retrieval area is limited, resulting in the omission of download. Therefore, if the download area is large, it is better to split it into several rectangles and download them one by one. Thus, a configuration parameter, `partition`, implements the sharding times of the retrieval area. If set to 2, the segmented rectangle is divided into four retrieval areas for download.


```python
bound_coordinate={'leftBottom':[108.776852,34.186027],'rightTop':[109.129275,34.382171]} 
page_num_range=range(20)
partition=3
query_dic={
    'query':'旅游景点',
    'page_size':'20',
    'scope':2,
    'ak':'uqRcWhrQ6h0pAaSdxYn73GMWgd5uNrRX',
}
```

Relying on the Baidu map open platform, POI data can be retrieved on the rectangular region by defining a function to facilitate the code transfer or invocation of related projects and increase the strength of code fusion.  Therefore, the input parameter must be carefully determined to avoid unnecessary errors caused by adjusting the variables within the function at the time of the call. The data downloaded from Baidu map, its latitude and longitude coordinates are Baidu coordinate system, so it needs to be converted. Append the conversion code `coordinate_transformation.py`(which comes from the network).

During the coordinate transformation, two functions are called `bd09togcj02(bd_lon, bd_lat)` and `gcj02towgs84(lng, lat)`. One way is to transfer directly, but because the tow functions also have dependent functions, calling the source makes the code structure clearer. The coordinate transformation file can be viewed from our GitHub hosted repository. When the crawler function is run, the transformation file needs to be placed in the same folder as the .ipynbn file to be called.


```python
import coordinate_transformation as cc
def baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False):
    '''function-Baidu map open platform POI data crawling'''
    urlRoot='http://api.map.baidu.com/place/v2/search?' #Data download site, query Baidu map service documents
    #The retrieval area sharding
    xDis=(bound_coordinate['rightTop'][0]-bound_coordinate['leftBottom'][0])/partition
    yDis=(bound_coordinate['rightTop'][1]-bound_coordinate['leftBottom'][1])/partition    
    #Determine whether to write to a file
    if poi_fn_list:
        for file_path in poi_fn_list:
            fP=pathlib.Path(file_path)
            if fP.suffix=='.csv':
                poi_csv=open(poi_fn_csv,'w',encoding='utf-8')
                csv_writer=csv.writer(poi_csv)    
            elif fP.suffix=='.json':
                poi_json=open(poi_fn_json,'w',encoding='utf-8')
    num=0
    jsonDS=[] #Store read data for the storage of .json formatted data
    #loop sharded retrieval area where data is downloaded block by block
    print("Start downloading data...")
    for i in range(partition):
        for j in range(partition):
            leftBottomCoordi=[bound_coordinate['leftBottom'][0]+i*xDis,bound_coordinate['leftBottom'][1]+j*yDis]
            rightTopCoordi=[bound_coordinate['leftBottom'][0]+(i+1)*xDis,bound_coordinate['leftBottom'][1]+(j+1)*yDis]
            for p in page_num_range:  
                #update request parameters
                query_dic.update({'page_num':str(p),
                                  'bounds':str(leftBottomCoordi[1]) + ',' + str(leftBottomCoordi[0]) + ','+str(rightTopCoordi[1]) + ',' + str(rightTopCoordi[0]),
                                  'output':'json',
                                 })
                
                url=urlRoot+urllib.parse.urlencode(query_dic)
                data=urllib.request.urlopen(url)
                responseOfLoad=json.loads(data.read()) 
                if responseOfLoad.get("message")=='ok':
                    results=responseOfLoad.get("results") 
                    for row in range(len(results)):
                        subData=results[row]
                        baidu_coordinateSystem=[subData.get('location').get('lng'),subData.get('location').get('lat')] #take Baidu coordinate system
                        Mars_coordinateSystem=cc.bd09togcj02(baidu_coordinateSystem[0], baidu_coordinateSystem[1]) #Baidu coordinate system -->Mars coordinate system
                        WGS84_coordinateSystem=cc.gcj02towgs84(Mars_coordinateSystem[0],Mars_coordinateSystem[1]) #Mars coordinate system-->WGS84
                        if csv_writer:
                            csv_writer.writerow([subData]) #line by line, write the .csv file
                        jsonDS.append(subData)
            num+=1       
            print("No."+str(num)+" was written to the .csv file.")
    if poi_json:       
        json.dump(jsonDS,poi_json)
        poi_json.write('\n')
        poi_json.close()
    if poi_csv:
        poi_csv.close()
    print("The download is complete.")
    return jsonDS
jsonDS=baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=[poi_fn_csv,poi_fn_json])    
```

    Start downloading data...
    No.1 was written to the .csv file.
    No.2 was written to the .csv file.
    No.3 was written to the .csv file.
    No.4 was written to the .csv file.
    No.5 was written to the .csv file.
    No.6 was written to the .csv file.
    No.7 was written to the .csv file.
    No.8 was written to the .csv file.
    No.9 was written to the .csv file.
    The download is complete.
    

### 1.2 Convert POI data in .csv format into DataFrame of the panda's library
Read the poi_csv.csv file that has been saved, because the csv and json libraries are used when the data is saved; the corresponding libraries are still used when they are read. The most commonly used pandas library also has methods such as `read_csv()` and `read_json()`, etc. However, there are various ways to store .csv files, so when reading .csv or .json file, it is best to read the file corresponding to the data format stored by itself. An error will occur when reading the .csv file of the POI saved above.  Only when you know the data format can you purposefully extract the data. The data format of each row is read as follows:
```python
["{'name': '昆明池遗址', 'location': {'lat': 34.210991, 'lng': 108.779778}, 'address': '西安市长安区昆明池七夕公园内', 'province': '陕西省', 'city': '西安市', 'area': '长安区', 'detail': 1, 'uid': 'c7332cd7fbcc0d82ebe582d9', 'detail_info': {'tag': '旅游景点;景点', 'navi_location': {'lng': 108.7812626866, 'lat': 34.217484892966}, 'type': 'scope', 'detail_url': 'http://api.map.baidu.com/place/detail?uid=c7332cd7fbcc0d82ebe582d9&amp;output=html&amp;source=placeapi_v2', 'overall_rating': '4.3', 'comment_num': '77', 'children': []}}"]
```
After reading .csv data, directly using .csv format data to extract and analyze data is not very convenient. The most commonly used data formats are array provided by Numpy library and DataFrame and Series provided by pandas. Among them, the data organization form of NumPy is more inclined to scientific calculation, in the form of number matrix, each array is the same data type. The DataFrame of pandas is similar to the attribute table in the geographic information data. Its column can be understood as a property field, and each column has the same data type so that a DataFrame can contain multiple data types. So for .csv data, the data format that converts it to DataFrame is the best. Furthermore, in the urban spatial data analysis method, more data based on geographic spatial location, so data with geographic attributes need to be processed with the idea of geographic information system. After being processed as DataFrame data format, GeoPandas and other geographic information libraries are further applied to deal with geospatial information data.

The raw data may have an incorrect data format, and errors often occur during data format conversion, such as the following inaccurate data:
```python
["{'name': '励进海升酒店-多功能厅', 'location': {'lat': 34.218525, 'lng': 108.891524}, 'address': '西安市高新区沣惠南路34号励进海升酒店4层', 'province': '陕西省', 'city': '西安市', 'a{'name': '红蚂蚁少儿创意美术馆'", " 'location': {'lat': 34.306666", " 'lng': 108.822465}", " 'address': '陕西省西安市未央区后围寨启航佳苑B区3层商铺'", " 'province': '陕西省'", " 'city': '西安市'", " 'area': '未央区'", " 'telephone': '18209227178", "15229372642'", " 'detail': 1", " 'uid': 'e3fd730bb528b40015c9050c'", " 'detail_info': {'tag': '文化传媒;美术馆'", " 'type': 'scope'", " 'detail_url': 'http://api.map.baidu.com/place/detail?uid=e3fd730bb528b40015c9050c&amp;output=html&amp;source=placeapi_v2'", " 'overall_rating': '0.0'", ' \'children\': []}}"']
```
`'a{'name': '红蚂蚁少儿创意美术馆'",` is an extra part, which does not conform to any data format. So when you do data processing, you need to respond to that. Typically handled with a `try/except` statement, it is best to return indexable information and process it to avoid data loss.


```python
import pandas as pd
from benedict import benedict #The benedict library, a subclass of dict that supports keylist/keypath, applies its flatten method to flatten the nested dictionary, ready for use with DataFrame data structure.
def csv2df(poi_fn_csv):
    '''function-Convert .csv format of POI data to DataFrame of pandas'''
    n=0
    with open(poi_fn_csv, newline='',encoding='utf-8') as csvfile:
        poi_reader=csv.reader(csvfile)
        poi_dict={}    
        poiExceptions_dict={}
        for row in poi_reader:    
            if row:
                try:
                    row_benedict=benedict(eval(row[0])) #Convert string dictionary "{}" to dictionary{} with the eval method
                    flatten_dict=row_benedict.flatten(separator='_') #Flatten the nested dictionary
                    poi_dict[n]=flatten_dict
                except:                    
                    print("incorrect format of data_row number:%s"%n)                    
                    poiExceptions_dict[n]=row
            n+=1
            #if n==5:break #Because of the number of cycles, when debugging the code, you can set the stop condition, saving time, and easy to view data.
    poi_df=pd.concat([pd.DataFrame(poi_dict[d_k].values(),index=poi_dict[d_k].keys(),columns=[d_k]).T for d_k in poi_dict.keys()], sort=True,axis=0)
    print("_"*50)
    for col in poi_df.columns:
        try:
            poi_df[col]=pd.to_numeric(poi_df[col])
        except:
            print("%s data type is not converted..."%(col))
    print("_"*50)
    print(".csv to DataFrame is completed!")
    #print(poi_df.head()) #View the POI data in the final DataFrame format
    #print(poi_df.dtypes) #view data types
    return poi_df
```

Almost all python data types, list, dictionary, set and class, and so on, can be serialized for storage by pickle, and pandas also provides `pandas.DataFrame.to_pickle` `pandas.DataFrame.read_pickle` to write. Therefore, to avoid converting .csv to DataFrame every time, the files of the pandas data types can be stored with the method provided by pandas.


```python
poi_df=csv2df(poi_fn_csv)
poi_fn_df=os.path.join(data_path,'poi_df.pkl')
poi_df.to_pickle(poi_fn_df)
print("_"*50)
print(poi_df.head())
```

    __________________________________________________
    address data type is not converted...
    area data type is not converted...
    city data type is not converted...
    detail_info_detail_url data type is not converted...
    detail_info_indoor_floor data type is not converted...
    detail_info_tag data type is not converted...
    detail_info_type data type is not converted...
    name data type is not converted...
    province data type is not converted...
    street_id data type is not converted...
    telephone data type is not converted...
    uid data type is not converted...
    __________________________________________________
    .csv to DataFrame is completed!
    __________________________________________________
               address area city  detail  detail_info_children  \
    0   西安市长安区昆明池七夕公园内  长安区  西安市       1                   NaN   
    2    外事学院南区北门内西50米  雁塔区  西安市       1                   NaN   
    4        西安市高新区云萃路  雁塔区  西安市       1                   NaN   
    6  西安市长安区昆明池·七夕公园内  长安区  西安市       1                   NaN   
    8     陕西省西安市雁塔区鱼跃路  雁塔区  西安市       1                   NaN   
    
       detail_info_comment_num                             detail_info_detail_url  \
    0                     77.0  http://api.map.baidu.com/place/detail?uid=c733...   
    2                      7.0  http://api.map.baidu.com/place/detail?uid=ae1c...   
    4                     32.0  http://api.map.baidu.com/place/detail?uid=6ccb...   
    6                      4.0  http://api.map.baidu.com/place/detail?uid=66f2...   
    8                      1.0  http://api.map.baidu.com/place/detail?uid=6e22...   
    
       detail_info_image_num detail_info_indoor_floor  \
    0                    NaN                      NaN   
    2                    NaN                      NaN   
    4                    NaN                      NaN   
    6                    1.0                      NaN   
    8                    NaN                      NaN   
    
       detail_info_navi_location_lat  ...  detail_info_price  detail_info_tag  \
    0                      34.217485  ...                NaN          旅游景点;景点   
    2                      34.239117  ...                NaN          旅游景点;公园   
    4                      34.219445  ...                NaN          旅游景点;公园   
    6                            NaN  ...                NaN          旅游景点;景点   
    8                            NaN  ...                NaN          旅游景点;公园   
    
       detail_info_type location_lat location_lng       name  province  \
    0             scope    34.210991   108.779778      昆明池遗址       陕西省   
    2             scope    34.237922   108.873804      鱼化湖公园       陕西省   
    4             scope    34.220518   108.846445       云水公园       陕西省   
    6             scope    34.218424   108.776999       汉武大帝       陕西省   
    8             scope    34.247621   108.844682  鱼化工业园亲水公园       陕西省   
    
                      street_id      telephone                       uid  
    0                       NaN            NaN  c7332cd7fbcc0d82ebe582d9  
    2  ae1cdccdcd8fdb6fb23c8188  (029)88751007  ae1cdccdcd8fdb6fb23c8188  
    4  6ccb87a451f19a27626858b9            NaN  6ccb87a451f19a27626858b9  
    6                       NaN            NaN  66f2005e22dcafdbc7f50d07  
    8                       NaN            NaN  6e22d8267dd611b4095cb38e  
    
    [5 rows x 22 columns]
    

### 1.3 POI spatial point map of a single classification 
#### 1.3.1 Convert POI data, whose data formate is DataFrame, into GeoDataFrame, a geospatial data format provided by GeoPandas library
GeoPandas can be used to transform the DataFrame and Series data of pandas into GeoDataFrame and GeoSeries geographic data with geographical significance. GeoPandas is based on pandas, so the most considerable difference in the data structure is that it has a column named 'geometry,' for storing geometric data, for example, `POINT (163.85316 -17.31631)`, `POLYGON ((33.90371 -0.95000, 31.86617 -1.02736...`, ` MULTIPOLYGON (((120.83390 12.70450, 120.32344 ...` etc... Note that the representation of geometric data is realized by using 'shapely' library, most of which are .shp geographic information vector data. In the python language, this library is commonly used as a base for building geometric objects. It should also be noted that the establishment of the GeoDataFrame object requires the configuration of the coordinate system, which is a prominent symbol of the geographic information data. It can also be obtained by viewing the coordinate system name from [spatialreference](https://spatialreference.org/ ).

GeoDataFrame displays geographical information data directly through .plot() method.


```python
poi_df=pd.read_pickle(poi_fn_df) #reads the POI in the saved .pkl(pickle) data format

import geopandas as gpd
from shapely.geometry import Point

poi_geoDF=poi_df.copy(deep=True)
poi_geoDF['geometry']=poi_geoDF.apply(lambda row:Point(row.location_lng,row.location_lat),axis=1) 
crs={'init': 'epsg:4326'} #coordinate-system configuration, reference：https://spatialreference.org/  
poi_gpd=gpd.GeoDataFrame(poi_geoDF,crs=crs)
poi_gpd.plot(column='detail_info_comment_num') #The argument shown here is set to column `detail_info_comment_num`, that is, the number of comments for tourist attractions in <'query':'旅游景点'>.
```
<a href="https://jupyter.org/"><img src="./imgs/expe_1_1.jpg" height="200" width="auto" title="caDesign">

#### 1.3.2 use the plotly library to create a map
Geopandas library provides a limited map display method, which is commonly used for data viewing because of its convenience. You can use the [plotly](https://plotly.com/) diagram library when you need a map of a certain quality, express more information, and even be interactive. Its background map uses map data provided by [mapbox](https://www.mapbox.com/). To utilize its functionality, you need to register and obtain an access token. This part of the operation is quite convenient. You can see it for yourself.

To create a map using the plotly library, you do not need to convert the DataFrame to GeoDataFrame.


```python
import plotly.express as px
poi_gpd.detail_info_price=poi_gpd.detail_info_price.fillna(0) #The approach to the pandas library is also applicable to geopandas library, for example, filling a specified number at 'nan' positions.
mapbox_token='pk.eyJ1IjoicmljaGllYmFvIiwiYSI6ImNrYjB3N2NyMzBlMG8yc254dTRzNnMyeHMifQ.QT7MdjQKs9Y6OtaJaJAn0A'
px.set_mapbox_access_token(mapbox_token)
fig=px.scatter_mapbox(poi_gpd,lat=poi_gpd.location_lat, lon=poi_gpd.location_lng,color="detail_info_comment_num",color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10) #You can also select columns, add display information by configuration size=""
fig.show()
```
<a href="https://jupyter.org/"><img src="./imgs/expe_1_2.jpg" height="auto" width="auto" title="caDesign">

### 1.4 key point
#### 1.4.1 data processing technique
* Update the data with a dictionary
Often parameters need to be configured, and if the parameter is not a fixed unique value, the value of the variable needs to be replaced. The dictionary `d.update()` method is an excellent choice for gathering the required parameters, increasing readability, and facilitating parameter updating


```python
query_dic={
    'query':'旅游景点',
    'page_size':'20',
    'scope':2,
    'ak':'uqRcWhrQ6h0pAaSdxYn73GMWgd5uNrRX',
}
query_dic.update({'output':'json'}) #add a new key and value
print(query_dic)
print("_"*50)
for p in range(2):  
    #update request parameters
    query_dic.update({'page_num':str(p),})#the original key and value are added the first time, and then the value of the key is updated.
    print(query_dic)
```

    {'query': '旅游景点', 'page_size': '20', 'scope': 2, 'ak': 'uqRcWhrQ6h0pAaSdxYn73GMWgd5uNrRX', 'output': 'json'}
    __________________________________________________
    {'query': '旅游景点', 'page_size': '20', 'scope': 2, 'ak': 'uqRcWhrQ6h0pAaSdxYn73GMWgd5uNrRX', 'output': 'json', 'page_num': '0'}
    {'query': '旅游景点', 'page_size': '20', 'scope': 2, 'ak': 'uqRcWhrQ6h0pAaSdxYn73GMWgd5uNrRX', 'output': 'json', 'page_num': '1'}
    

* use the Path method provided by pathlib library to set up the path and extract the relevant attributes concisely

The operation of the path mainly uses three libraries, one is the classic os, the other si the pathlib, and the third is the glob.


```python
import os,pathlib
dir_path=os.getcwd()+"\BaiduMapPOI_collection_singleClassification.ipynb"
print(dir_path)
pb_path=pathlib.Path(dir_path)
print(
    "1_anchor "+pb_path.anchor+"\n",
    "2_drive "+pb_path.drive+"\n",
    "3_name "+pb_path.name+"\n",
    #"4_parent "+pb_path.parent+"\n",
    #"5_parents "+pb_path.parents+"\n",
    #"6_parts "+pb_path.parts+"\n",
    "7_root "+pb_path.root+"\n",
    "8_stem "+pb_path.stem+"\n",
    "9_suffix "+pb_path.suffix+"\n",
    #"10_suffixes "+pb_path.suffixes+"\n",
    )
```

    C:\Users\richi\omen-richiebao\omen_github\Urban-Spatial-Data-Analysis_python\notebook\BaiduMapPOIcollection_ipynb\BaiduMapPOI_collection_singleClassification.ipynb
    1_anchor C:\
     2_drive C:
     3_name BaiduMapPOI_collection_singleClassification.ipynb
     7_root \
     8_stem BaiduMapPOI_collection_singleClassification
     9_suffix .ipynb
    
    

* There are three ways to save files: csv,json, and pickle in this experiment.

`csv_writer=csv.writer()->csv_writer.writerow([]) csv.reader()`，`json.dump() json.loads`，`pickle.dumps() pickle.loads()`

* Use the benedict library to flatten the nested dictionary.
In this experiment, after reading the POI data in .csv format, the data obtained is a nested dictionary, which is in string format, so the string dictionary is first converted into a dictionary using the eval() method, and then flattened out.


```python
d_nested="{'name': '昆明池遗址', 'location': {'lat': 34.210991, 'lng': 108.779778}, 'address': '西安市长安区昆明池七夕公园内', 'province': '陕西省', 'city': '西安市', 'area': '长安区', 'detail': 1, 'uid': 'c7332cd7fbcc0d82ebe582d9', 'detail_info': {'tag': '旅游景点;景点', 'navi_location': {'lng': 108.7812626866, 'lat': 34.217484892966}, 'type': 'scope', 'detail_url': 'http://api.map.baidu.com/place/detail?uid=c7332cd7fbcc0d82ebe582d9&amp;output=html&amp;source=placeapi_v2', 'overall_rating': '4.3', 'comment_num': '77', 'children': []}}"
from benedict import benedict
d_benedict=benedict(eval(d_nested))
d_flatten=d_benedict.flatten(separator='_')
print(d_flatten)
```

    {'name': '昆明池遗址', 'location_lat': 34.210991, 'location_lng': 108.779778, 'address': '西安市长安区昆明池七夕公园内', 'province': '陕西省', 'city': '西安市', 'area': '长安区', 'detail': 1, 'uid': 'c7332cd7fbcc0d82ebe582d9', 'detail_info_tag': '旅游景点;景点', 'detail_info_navi_location_lng': 108.7812626866, 'detail_info_navi_location_lat': 34.217484892966, 'detail_info_type': 'scope', 'detail_info_detail_url': 'http://api.map.baidu.com/place/detail?uid=c7332cd7fbcc0d82ebe582d9&amp;output=html&amp;source=placeapi_v2', 'detail_info_overall_rating': '4.3', 'detail_info_comment_num': '77', 'detail_info_children': []}
    

* Converts the nested dictionary to the DataFrame data format

use the above-mentioned flattened dictionary directly


```python
import pandas as pd
d_df=pd.DataFrame(d_flatten.values(),index=d_flatten.keys(),columns=["val"])
print(d_df.head())
```

                             val
    name                   昆明池遗址
    location_lat          34.211
    location_lng          108.78
    address       西安市长安区昆明池七夕公园内
    province                 陕西省
    

* geopandas、gdal、ogr、rasterstats、pysal and other python geographic information libraries assist in analyzing geographic information data. 'shapely' is used to process geometric data.

#### 1.4.2 The newly created function tool
* function-Baidu map open platform POI data crawler，`baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False)`;
* function-Convert .csv format of POI data to DataFrame of pandas,`csv2df(poi_fn_csv)`。

#### 1.4.3 The python libraries that are being imported


```python
import urllib, json, csv,os,pathlib
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from benedict import benedict

import plotly.express as px
from plotly.offline import plot
```