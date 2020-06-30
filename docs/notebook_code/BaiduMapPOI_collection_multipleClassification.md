> Created on Thu Nov 16 12/12/38 2017  @author: Richie Bao-caDesign设计(cadesign.cn)
> __+updated on Fri Jun 26 11/02/19 2020 by Richie Bao
## 1. Multiple classification POI data crawling and Descriptive statistics
[JupterLab .ipynb file download location](https://github.com/richieBao/Urban-Spatial-Data-Analysis_python_EN/tree/master/notebook/BaiduMapPOIcollection_ipynb)

### 1.1 Multiple classification POI data crawling 
In the previous section, two new function tools were established, namely 'Baidu map open platform POI data crawler' and 'Convert .csv format of POI data to DataFrame of pandas.' To make it easier to apply the function tool, use Anaconda's Spyder to create a new file, 'util_poi.py,' to put the two functions into it, along with the libraries used by the function.  For the included libraries, the statements corresponding to the calling libraries are placed inside each function to facilitate future function transfer and to specify which libraries each function calls.  'util_poi.py' is in the same folder as the file to be called.  The 'util_poi.py' file is available for download from the GitHub repository. The call statement is as follows:

```python
import util_poi
```

According to the Baidu map's first-level industry classification, a mapping dictionary was established for data crawling of multiple POI classification. It should be noted that the industry classification can be selected according to the data requirements. The categories not needed in the further analysis include 'entrance and exit,' 'natural feature,' 'landmark', and 'door address,' so the above classification is not included in the mapping dictionary.

```python
poi_classificationName={
        "美食 ":"delicacy",
        "酒店 ":"hotel",
        "购物 ":"shopping",
        "生活服务":"lifeService",
        "丽人 ":"beauty",
        "旅游景点":"spot",
        "休闲娱乐":"entertainment",
        "运动健身":"sports",
        "教育培训":"education",
        "文化传媒":"media",
        "医疗 ":"medicalTreatment",
        "汽车服务":"carService",
        "交通设施":"trafficFacilities",
        "金融":"finance",
        "房地产":"realEstate",
        "公司企业":"corporation",
        "政府机构":"government"
        }
```

Configure the base parameters. Note the `query_dic` was used in the previous section. This time, all parameters are given separately in a dictionary from outside the loop function for easy invocation. The `query_dic` dictionary parameters are configured within the bulk download function.

This part of the code is only used to illustrate the parameter configuration form in the previous section. No typing is required this time.
```
query_dic={
    'query':'旅游景点',
    'page_size':'20',
    'scope':2,
    'ak':'uqRcWhrQ6h0pAaSdxYn73GMWgd5uNrRX',
}
```

```python
poi_config_para={
    'data_path':'./data/poi_batchCrawler/', #configure the data storage location
    'bound_coordinate':{'leftBottom':[108.776852,34.186027],'rightTop':[109.129275,34.382171]}, #Baidu map coordinate pickup system  http://api.map.baidu.com/lbsapi/getpoint/index.html
    'page_num_range':range(20),
    'partition':3, #3
    'page_size':'20', #20
    'scope':2,
    'ak':'uqRcWhrQ6h0pAaSdxYn73GMWgd5uNrRX',
}
```

Set up the loop function of batch download, and invoke the single POI classification successively according to the given `poi_classificationName` dictionary key-value. In the process of crawling, each small batch of downloaded data can be stored in the same variable and saved once after all downloads. However, this one-time storage method is not recommended. Firstly, the network is sometimes unstable, which may cause the interruption of download. Then the downloaded data cannot be stored, resulting in data loss and unnecessarily repeated downloading.  Second, sometimes the data volume is vast if all stored under a variable may cause memory overflow.

```python
def baiduPOI_batchCrawler(poi_config_para):
    import os
    import util_poi
    '''function-Baidu Map open platform POI data mass crawling, need to call the single POI classification crawler function. baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False)'''
    for idx,(poi_ClassiName,poi_classMapping) in enumerate(poi_classificationName.items()):
        print(str(idx)+"_"+poi_ClassiName)
        poi_subFileName="poi_"+str(idx)+"_"+poi_classMapping
        data_path=poi_config_para['data_path']
        poi_fn_csv=os.path.join(data_path,poi_subFileName+'.csv')
        poi_fn_json=os.path.join(data_path,poi_subFileName+'.json')
        
        query_dic={
            'query':poi_ClassiName,
            'page_size':poi_config_para['page_size'],
            'scope':poi_config_para['scope'],
            'ak':poi_config_para['ak']                        
        }
        bound_coordinate=poi_config_para['bound_coordinate']
        partition=poi_config_para['partition']
        page_num_range=poi_config_para['page_num_range']
        #call the single POI classification crawler function
        util_poi.baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=[poi_fn_csv,poi_fn_json])  
        
baiduPOI_batchCrawler(poi_config_para)
```

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
0_美食 
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
1_酒店 
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
2_购物 
Start downloading data...
No.1 was written to the .csv file.
No.2 was written to the .csv file.
No.3 was written to the .csv file.
```

### 1.2 Batch conversion .csv format data to GeoDataFrame
In the single classification implementation part, we are step by step to implement .csv format data to GeoDataFrame data conversion. Based on the existing code, two purposes will be achieved in this section. One is to define a separate function implementation .csv bulk conversion to GeoDataFame format data and save as .pkl file. The second is to read the data in the format of GeoDataFrame stored as .pkl file in batches, and extract the information as needed and then save it as .pkl file.  It should be noted that when all data is read in a single variable, the memory needs to meet the requirements. If there is a memory overflow, it needs to consider whether to adjust the amount of data read each time according to the memory situation.
#### 1.2.1 define a function to extract all file paths under a folder
Since the POI data is downloaded in batches as multiple .csv and .json format files, the first thing to do when processing the data in batches is to extract all the files' paths. The function that returns the file paths of all the specified suffix names is one of the most commonly used functions that will be called in many experiments, so you can also save it in the 'util_poi.py' file for later invocation. At the same time, it should be noted that the folder usually includes subfolders, and `os.walk()` is required to traverse the directory. A conditional statement is given to determine whether there is a subfolder, if there is, the file path under the folder needs to be returned. 

```python
def filePath_extraction(dirpath,fileType):
    import os
    '''funciton-With the folder path as the key, the value is a list of all filenames under the folder. File types can be specified.  '''
    filePath_Info={}
    i=0
    for dirpath,dirNames,fileNames in os.walk(dirpath): #'os.walk()' walks through the directory, using 'help(os.walk)' to find the explanation of the return values. 
       i+=1
       if fileNames: #extract file path only if there are files in the folder
           tempList=[f for f in fileNames if f.split('.')[-1] in fileType]
           if tempList: #If the filename list is empty, if there are files under the folder that are not of the specified file type, the previous list will return an empty list [].
               filePath_Info.setdefault(dirpath,tempList)
    return filePath_Info

dirpath='./data/poi_batchCrawler/'
fileType=["csv"]
poi_paths=filePath_extraction(dirpath,fileType)
print(poi_paths)
```

```
{'./data/poi_batchCrawler/': ['poi_0_delicacy.csv', 'poi_10_medicalTreatment.csv', 'poi_11_carService.csv', 'poi_12_trafficFacilities.csv', 'poi_13_finance.csv', 'poi_14_realEstate.csv', 'poi_15_corporation.csv', 'poi_16_government.csv', 'poi_1_hotel.csv', 'poi_2_shopping.csv', 'poi_3_lifeService.csv', 'poi_4_beauty.csv', 'poi_5_spot.csv', 'poi_6_entertainment.csv', 'poi_7_sports.csv', 'poi_8_education.csv', 'poi_9_media.csv']}
```

#### 1.2.2 convert .csv format POI data to GeoDataFrame in bulk
```
Index(['address', 'area', 'city', 'detail', 'detail_info_checkin_num','detail_info_children', 'detail_info_comment_num',
       'detail_info_detail_url', 'detail_info_facility_rating','detail_info_favorite_num', 'detail_info_hygiene_rating',
       'detail_info_image_num', 'detail_info_indoor_floor','detail_info_navi_location_lat', 'detail_info_navi_location_lng',
       'detail_info_overall_rating', 'detail_info_price','detail_info_service_rating', 'detail_info_tag', 'detail_info_type',
       'location_lat', 'location_lng', 'name', 'province', 'street_id','telephone', 'uid'],dtype='object')
```
The above are POI data fields, based on which you can determine the extracted field name.  In addition to adding a loop statement to convert .csv file to DataFrame format file one by one, further conversion to the GeoDataFrame format defined by GeoPandas'; all other conditions are the same as in the previous section. 'GeoDataFrame.plot()' gives a direct preliminary view of the geospatial information data. Under file saving, you can choose from a variety of saving formats, pickle, and GeoPandas offers Shapefile、GeoJSON , 0GeoPackage offered by GeoPandas. GeoPandas offers the saving format that no longer includes multiple indexes when read, while the pickle format remains. When converting to .shp format file, there will be two problems when it is opened under QGIS and other desktop GIS platform. One is that if the column name is too long, the field name converted to the attribute table will be compressed and modified, and often cannot reflect the meaning of the field, so the column names need to be replaced. Second, when the POI's first-level industry classification name is used as an index, the column does not contain this field, nor does it contains this field when converted to Shapefile. So you need to convert the index to a column and save it as a .shp file.


```python
fields_extraction=['name','location_lat', 'location_lng','detail_info_tag','detail_info_overall_rating', 'detail_info_price'] configure the fields to be extracted, namely columns
save_path={'geojson':'./data/poiAll_gpd.geojson','shp':'./data/poiAll_gpd.shp','pkl':'./data/poiAll_gpd.pkl'} #They are saved in three data formats: GeoJSON、Shapefile and pickle.
def poi_csv2GeoDF_batch(poi_paths,fields_extraction,save_path):
    import os,pathlib
    import util_poi
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point
    '''funciton-.csv format  POI data is converted to GeoDataFrame in batch, which needs to call .csv to DataFrame data format provided by pandas function `csv2df(poi_fn_csv)`.'''
    #cyclic read and convert .csv POI data to DataFrame data format provided by pandas
    poi_df_dic={}
    i=0
    for key in poi_paths:
        for val in poi_paths[key]:
            poi_csvPath=os.path.join(key,val)
            poi_df=util_poi.csv2df(poi_csvPath) #comment out the statement `print("%s data type is not converted..."%(col))` under the function `csv2df()`, replaced by `pass`, to reduce the prompt content, avoid interference. 
            print(val)
            poi_df_path=pathlib.Path(val)
            poi_df_dic[poi_df_path.stem]=poi_df
            
            #if i==2:break
            i+=1
    poi_df_concat=pd.concat(poi_df_dic.values(),keys=poi_df_dic.keys(),sort=True)
    #print(poi_df_concat.loc[['poi_0_delicacy'],:]) #extract the line of index  'poi_0_delicacy' and verify the result
    poi_fieldsExtraction=poi_df_concat.loc[:,fields_extraction]
    poi_geoDF=poi_fieldsExtraction.copy(deep=True)
    poi_geoDF['geometry']=poi_geoDF.apply(lambda row:Point(row.location_lng,row.location_lat),axis=1) 
    crs={'init': 'epsg:4326'} #coordinate-system configuration, reference:https://spatialreference.org/        
    poiAll_gpd=gpd.GeoDataFrame(poi_geoDF,crs=crs)
    
    poiAll_gpd.to_pickle(save_path['pkl'])
    poiAll_gpd.to_file(save_path['geojson'],driver='GeoJSON')
    
    poiAll_gpd2shp=poiAll_gpd.reset_index() #Not specifying a level parameter, such as level=0, converts all indexes in multiple indexes to columns.
    poiAll_gpd2shp.rename(columns={
        'location_lat':'lat', 'location_lng':'lng',
        'detail_info_tag':'tag','detail_info_overall_rating':'rating', 'detail_info_price':'price'},inplace=True)
    poiAll_gpd2shp.to_file(save_path['shp'],encoding='utf-8')
        
    return poiAll_gpd
            
poi_gpd=poi_csv2GeoDF_batch(poi_paths,fields_extraction,save_path)
poi_gpd.loc[['poi_0_delicacy'],:].plot(column='detail_info_overall_rating') #extract the row of index 'poi_0_delicacy' to view the results
```

```
__________________________________________________
.csv to DataFrame is completed!
poi_0_delicacy.csv
__________________________________________________
.csv to DataFrame is completed!
poi_10_medicalTreatment.csv
__________________________________________________
.csv to DataFrame is completed!
poi_11_carService.csv
__________________________________________________
.csv to DataFrame is completed!
poi_12_trafficFacilities.csv
__________________________________________________
.csv to DataFrame is completed!
poi_13_finance.csv
__________________________________________________
.csv to DataFrame is completed!
poi_14_realEstate.csv
__________________________________________________
.csv to DataFrame is completed!
poi_15_corporation.csv
__________________________________________________
.csv to DataFrame is completed!
poi_16_government.csv
__________________________________________________
```

<a href="https://jupyter.org/"><img src="./imgs/2_1.png" height="auto" width="auto" title="caDesign"/></a>

> Open saved .shp data under [QGIS open-source desktop GIS platform](https://www.qgis.org/en/site/index.html) . While almost all of our work is done in Python, some work is linked to other platforms and need to be aided by those platforms. In the selection of auxiliary platforms, we use open source software with a wide application as far as possible. QGIS and ArcGIS are the main integrated platforms used in Geographic Information Systems.

<a href=""><img src="./imgs/2_2.jpg" height="auto" width="auto" title="caDesign"/>

### 1.3 Use the Plotly library to create a map.
The POI primary classification is represented by color and the rating field by size.

```python
import geopandas as gpd
poi_gpd=gpd.read_file('./data/poiAll_gpd.shp') #Read a saved .shp file

import plotly.express as px
poi_gpd.rating=poi_gpd.rating.fillna(0) #The Pandas library approach is also applicable to the GeoPandas library, for example, filling a specified number of 'nan' positions.
mapbox_token='pk.eyJ1IjoicmljaGllYmFvIiwiYSI6ImNrYjB3N2NyMzBlMG8yc254dTRzNnMyeHMifQ.QT7MdjQKs9Y6OtaJaJAn0A'
px.set_mapbox_access_token(mapbox_token)
fig=px.scatter_mapbox(poi_gpd,lat=poi_gpd.lat, lon=poi_gpd.lng,color="level_0",size='rating',color_continuous_scale=px.colors.cyclical.IceFire, size_max=10, zoom=10) #You can also select columns to increase display information by configuring size=''.
fig.show()
```

<a href=""><img src="./imgs/2_3.jpg" height="auto" width="auto" title="caDesign"/></a>

### 1.4 Descriptive statistics
#### 1.4.1 read data and view
* Read the saved .pkl data. .plot() to ensure that the read data is normal, or directly `poi_gpd.head()` to view the data.

```python
import pandas as pd
poi_gpd=pd.read_pickle('./data/poiAll_gpd.pkl')
poi_gpd.plot(marker=".",markersize=5,column='detail_info_overall_rating') #Only if you do not set the parameters of the column, you can use the color='green' parameter.
print(poi_gpd.columns) #view column names
```

```
Index(['name', 'location_lat', 'location_lng', 'detail_info_tag',
       'detail_info_overall_rating', 'detail_info_price', 'geometry'],
      dtype='object')
```

<a href=""><img src="./imgs/2_4.png" height="auto" width="auto" title="caDesign"/></a>

#### 1.4.2 Display the DataFrame data with the Plotly table.

'print()' is the primary way to view the data and mainly used for code debugging. When data needs to be presented, data in DataFrame format can be converted directly using Ployly to tabular form, because the POI data is as many as 10,000 rows, showing only the first two rows for each first-level industry classification (with 17 classes), for a total of $2\times17=34$ rows. Its function is defined as a function for ease of invocation. When you extract the data, because the data format is multi-index DataFrame, you use the `pandas.IndexSlice()` function to assist with multi-index sharding. When displaying the table with Plotly at the same time, an error will be displayed if there are multiple indexes, so `df.reset_index()` is required to reset the index. Plotly also cannot show 'geometry' objects and needs to be removed when the columns are extracted.

```python
df=poi_gpd.loc[pd.IndexSlice[:,:2],:]
df=df.reset_index()
column_extraction=['level_0','name', 'location_lat', 'location_lng', 'detail_info_tag','detail_info_overall_rating', 'detail_info_price']

def ployly_table(df,column_extraction):
    import plotly.graph_objects as go
    import pandas as pd
    '''funciton-use Plotly to display data in DataFrame format as a table'''
    fig = go.Figure(data=[go.Table(
        header=dict(values=column_extraction,
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=df[column_extraction].values.T.tolist(), #The values parameter is a nested list by column, so you need to use the `.T` to reverse the array.
                   fill_color='lavender',
                   align='left'))
    ])
    fig.show()    
ployly_table(df,column_extraction)
```

<a href=""><img src="./imgs/2_5.jpg" height="auto" width="auto" title="caDesign"/></a>

#### 1.4.3 Descriptive statistics

> Reference: Shin Takahashi.2008.The Manga Guide to Statistics. No Starch Press; 1 edition (November 15, 2008). and Wikipedia. The annoying knowledge is told in the way of a cartoon and combined with practical cases; boring learning becomes more attractive from simple to complicated. The Ohmsha learning comic series and numerous excellent books that illustrate knowledge in comics and illustrations are recommended. However, there are pros and cons. Most comic books focus on basic knowledge, and in-depth research still requires searching for scientific literature and relevant works. Simultaneously, funny forms can arouse the reader's interest. Still, because of the interspersed story, the knowledge points are not easy to locate, and the core knowledge content is relatively scattered, more time will be spent on reading. So want to learn a piece of knowledge, want to start with which form, and determine according to the individual situation.

Descriptive statistical analysis is a statistical description of the relevant data of all variables in the survey population to understand the concentration and dispersion of observed values in each variable.

-Represent a central tendency: mean, median, mode, geometric mean, harmonic mean, etc. Represent statistical dispersion(variation): range, mean deviation, standard deviation, relative difference, four-point difference, etc. The frequency distribution of data usually presents a normal distribution.

-The frequency distribution of data tends to be normally distributed. Skewness and kurtosis are used to present the deviation of the measured data from the normal distribution.

-Observations need to be converted into relative quantities, such as percentiles, standard fractions, quartiles, etc., to understand the individual's position as a whole.

Generally, the data are gathered in descriptive statistics to understand the overall data distribution in an intuitive way, including histogram, scatter plot, pie chart, line chart, box chart, etc. 


##### 1. data types
Generally, the data can be divided into two categories. The non-measurable data is called classified data. Measurable data is called numerical data. The 'level_0', 'detail_info_tag' in the above chart are all classified data, 'location_lat', 'location_lng','detail_info_overall_rating', 'detail_info_price' are all numerical data. The 'name' field is the index name of the data.

##### 2. descriptive statistics of numerical data

###### frequency distribution and histogram

🍅 A simple data example was established based on the prices of ramen noodle restaurants published on the top 50 best-sellers of Delicious Ramen in The Manga Guide to Statistics. Although a set of data is typically created in Series format using `pandas.Series()`, DataFrame format data is still created because the subsequent analysis will add new data. You can look at the main statistics roughly using `df.describe().`

```python
ranmen_price=pd.DataFrame([700,850,600,650,980,750,500,890,880,700,890,720,680,650,790,670,680,900,880,720,850,700,780,850,750,
                           780,590,650,580,750,800,550,750,700,600,800,800,880,790,790,780,600,690,680,650,890,930,650,777,700],columns=["price"])
print(ranmen_price.describe())
```

```

            price
count   50.000000
mean   743.340000
std    108.261891
min    500.000000
25%    672.500000
50%    750.000000
75%    800.000000
max    980.000000
```

Since some prices are the same, the above `ranmen_price` data can be used to calculate the frequency directly. However, in many cases, the same data is not enough, and the analysis is more meaningful if the analysis content is the comparison between numerical extents. Therefore, it is converted into relative quantities with 1000 intervals as a level. The range is determined by the maximum and minimum values of the data.

```python
bins=range(500,1000+100,100) #configure split intervals (group spacing)
ranmen_price['price_bins']=pd.cut(x=ranmen_price.price,bins=bins,right=False) #The parameter `right=False` is specified to include the left value and not the right value.
ranmenPrice_bins=ranmen_price.sort_values(by=['price']) #sort by group spacing
ranmenPrice_bins.set_index(['price_bins',ranmenPrice_bins.index],drop=False,inplace=True) #Multiple indexes are set with `price_bins` and the original index value, and the `drop=False` parameter is configured to leave the original column.
print(ranmen_price.head(10))
```

```
   price   price_bins
0    700   [700, 800)
1    850   [800, 900)
2    600   [600, 700)
3    650   [600, 700)
4    980  [900, 1000)
5    750   [700, 800)
6    500   [500, 600)
7    890   [800, 900)
8    880   [800, 900)
9    700   [700, 800)
```

Frequency calculation

```python
ranmenPriceBins_frequency=ranmenPrice_bins.price_bins.value_counts() #dropna=False  
ranmenPriceBins_relativeFrequency=ranmenPrice_bins.price_bins.value_counts(normalize=True) #The parameter `normalize=True` will calculate the relative frequency. dividing all values by the sum of values
ranmenPriceBins_freqANDrelFreq=pd.DataFrame({'fre':ranmenPriceBins_frequency,'relFre':ranmenPriceBins_relativeFrequency})
print(ranmenPriceBins_freqANDrelFreq)
```

```
             fre  relFre
[700, 800)    18    0.36
[600, 700)    13    0.26
[800, 900)    12    0.24
[500, 600)     4    0.08
[900, 1000)    3    0.06
```

The group median value is calculated.

```python
ranmenPriceBins_median=ranmenPrice_bins.median(level=0)
ranmenPriceBins_median.rename(columns={'price':'median'},inplace=True)
print(ranmenPriceBins_median)
```

```
             median
price_bins         
[500, 600)      565
[600, 700)      650
[700, 800)      750
[800, 900)      865
[900, 1000)     930
```


Combine group spacing, frequency, and the media of the group data in DataFrame format.

```python
ranmen_fre=ranmenPriceBins_freqANDrelFreq.join(ranmenPriceBins_median).sort_index().reset_index() #The index is automatically matched when merging.
print(ranmen_fre)
```

```
         index  fre  relFre  median
0   [500, 600)    4    0.08     565
1   [600, 700)   13    0.26     650
2   [700, 800)   18    0.36     750
3   [800, 900)   12    0.24     865
4  [900, 1000)    3    0.06     930
```

Calculating the frequency ratio, that is, the percentage of frequencies in each interval as a percentage of the total provides a more explicit comparison of the differences.  Together with the use of `df.apply()` and the anonymous function, lambda, these two methods are often used to solve related problems neatly and concisely.

```python
ranmen_fre['fre_percent%']=ranmen_fre.apply(lambda row:row['fre']/ranmen_fre.fre.sum()*100,axis=1)
print(ranmen_fre)
```

```
         index  fre  relFre  median  fre_percent  fre_percent%
0   [500, 600)    4    0.08     565          8.0           8.0
1   [600, 700)   13    0.26     650         26.0          26.0
2   [700, 800)   18    0.36     750         36.0          36.0
3   [800, 900)   12    0.24     865         24.0          24.0
4  [900, 1000)    3    0.06     930          6.0           6.0
```


Histogram(pandas). Pandas comes with a lot of chart printing(based on the Matplotlib library) to quickly preview the data without tweaking the data structure too much. But unlike Plotly, there is no interaction.

```python
ranmen_fre.loc[:,['fre','index']].plot.bar(x='index',rot=0,figsize=(5,5))
```

<a href=""><img src="./imgs/2_6.png" height="auto" width="auto" title="caDesign"/></a>

🐨By obtaining the sample of the above simple data and returning to the POI experimental data, we could directly transfer the above code and analyze the overall price distribution of a first-class classification, delicacy, 'poi_0_delicacy' in POI, after some adjustments. Place all of the above analysis data into a function whose primary function is to calculate the frequency of column values of data in the DataFrame format by specifying the group spacing. To include the above trivial code into a function, we need to pay more attention to several things. One is to make the variable name as general as possible. For example, the original variable name `ranmenPrice_bins` changes to `df_bins` in the function, because this function can also calculate the frequency distribution of the 'detail_info_overall_rating' field. Secondly, common variables such as `column_name` and `column_bins_name` should be started as far as possible, to avoid using the original statements regularly, such as repeating `df.columns[0]+'_bins'`, resulting in poor code readability.
Moreover, it is almost impossible to transfer the above code directly for use in the function, and a sentence by sentence or segment transfer is required. For example, the original code `ranmenPrice_bins.price_bins` is changed into `df_bins[column_bins_name` in the function, because the column names are stored in the form of variable names, and the data can not be read directly in the way of `.`.  Finally is the need to pay attention to the flexibility of the return value by the function, such as has not defined the chart print within the function, but returns the DataFrame format data, because of the way to print more diversification, the histogram can only print a list of data or more columns, as well as charts will also be in the form of diversification, so this part of the task place to the function outside processing, increase flexibility.

```python
delicacy_price=poi_gpd.xs('poi_0_delicacy',level=0).detail_info_price #The food classification is extracted by multiple indexes, and further, the price column data is retrieved. 
delicacy_price_df=delicacy_price.to_frame(name='price')
print(delicacy_price_df.describe())
```

```
            price
count  785.000000
mean    53.584076
std     44.123529
min      5.000000
25%     23.000000
50%     43.000000
75%     72.000000
max    571.000000
```

```python
def frequency_bins(df,bins):
    import pandas as pd
    '''function-frequency calculation'''
    
    #A-organize data
    column_name=df.columns[0]
    column_bins_name=df.columns[0]+'_bins'
    df[column_bins_name]=pd.cut(x=df[column_name],bins=bins,right=False) #The parameter `right=False` is specified to include the left value and not the right value.
    df_bins=df.sort_values(by=[column_name]) #sort by group spacing
    df_bins.set_index([column_bins_name,df_bins.index],drop=False,inplace=True) #Multiple indexes are set with `price_bins` and the original index value, and the `drop=False` parameter is configured to leave the original column.
    #print(df_bins.head(10))
    
    #B-frequency calculation
    dfBins_frequency=df_bins[column_bins_name].value_counts() #dropna=False  
    dfBins_relativeFrequency=df_bins[column_bins_name].value_counts(normalize=True) #The parameter `normalize=True` will calculate the relative frequency. dividing all values by the sum of values
    dfBins_freqANDrelFreq=pd.DataFrame({'fre':dfBins_frequency,'relFre':dfBins_relativeFrequency})
    #print(dfBins_freqANDrelFreq)
    
    #C-the median value of the group calculation
    dfBins_median=df_bins.median(level=0)
    dfBins_median.rename(columns={column_name:'median'},inplace=True)
    #print(dfBins_median)
    
    #D-Combine group spacing, frequency, and the media of the group data in DataFrame format.
    df_fre=dfBins_freqANDrelFreq.join(dfBins_median).sort_index().reset_index() #The index is automatically matched when merging.
    #print(ranmen_fre)
    
    #E-Calculating the frequency ratio
    df_fre['fre_percent%']=df_fre.apply(lambda row:row['fre']/df_fre.fre.sum()*100,axis=1)
    
    return df_fre
bins=range(0,600+50,50) #configure split intervals (group spacing)    
poiPrice_fre_50=frequency_bins(delicacy_price_df,bins)    
print(poiPrice_fre_50)
```

```
         index  fre    relFre  median  fre_percent%
0      [0, 50)  445  0.566879    26.0     56.687898
1    [50, 100)  257  0.327389    70.0     32.738854
2   [100, 150)   63  0.080255   110.0      8.025478
3   [150, 200)   12  0.015287   165.5      1.528662
4   [200, 250)    4  0.005096   214.5      0.509554
5   [250, 300)    1  0.001274   285.0      0.127389
6   [300, 350)    0  0.000000     NaN      0.000000
7   [350, 400)    2  0.002548     NaN      0.254777
8   [400, 450)    0  0.000000   571.0      0.000000
9   [450, 500)    0  0.000000     NaN      0.000000
10  [500, 550)    0  0.000000     NaN      0.000000
11  [550, 600)    1  0.001274   381.0      0.127389
```

```python
poiPrice_fre_50.loc[:,['fre','index']].plot.bar(x='index',rot=0,figsize=(15,5))
```

<a href=""><img src="./imgs/2_7.png" height="auto" width="auto" title="caDesign"/></a>

Adjust the group spacing to see the overall frequency distribution.

```python
bins=list(range(0,300+5,5))+[600] #After viewing the data through `df.describe()`, it was found that 72% of the price was below 70 yuan. In combination with the bar chart above, the group spacing was reconfigured to show the trend of data changes as far as possible.
poiPrice_fre_5=frequency_bins(delicacy_price_df,bins)    
import matplotlib.pyplot as plt
poiPrice_fre_5.loc[:,['fre','index']].plot.bar(x='index',rot=0,figsize=(30,5))
plt.xticks(rotation=90)
```

<a href=""><img src="./imgs/2_8.png" height="auto" width="auto" title="caDesign"/></a>

Generally, things are normally distributed. When the group spacing is about 5, such a data structure or rules of things will appear, and the trend of price changes can also be found.

###### central tendency and variation

🍅Create a simple data example based on the bowing contest results in The Manga Guide to Statistics. The nested dictionary is first created and then converted to a DataFrame format with multiple indexes.

```python
bowlingContest_scores_dic={'A_team':{'Barney':86,'Harold':73,'Chris':124,'Neil':111,'Tony':90,'Simon':38},
                            "B_team":{'Jo':84,'Dina':71,'Graham':103,'Joe':85,'Alan':90,'Billy':89},
                            'C_team':{'Gordon':229,'Wade':77,'Cliff':59,'Arthur':95,'David':70,'Charles':88}
                          }
bowlingContest_scores=pd.DataFrame.from_dict(bowlingContest_scores_dic, orient='index').stack().to_frame(name='score') #The data structure of each step can be disassembled step by step, and the function of each step can be understood in combination with the explanation of searching related methods. For example, `df.stack()` returns the DataFrame indexed by columns. You can see the official case for a more intuitive understanding of this role.
bowlingContest_scores #Use `print()` or directly in each cell to give the variable to view the data. Still, the mode may display slightly different. `print()`  is recommended, because the presence of individual variables can cause code runtime errors when it comes to code transfer. 
```

<a href=""><img src="./imgs/2_9.jpg" height="500" width="auto" title="caDesign"/></a>

find the mean of each team (arithmetic mean)

```python
bowlingContest_mean=bowlingContest_scores.mean(level=0)
print(bowlingContest_mean)
```

```
        score
A_team   87.0
B_team   87.0
C_team  103.0
```

Find the median of each column. Team C had the highest average because not everyone had a high score, but because Gordon scored 229 points, which was much higher than the rest of the ream. So it is better to find the median.

```python
bowlingContest_median=bowlingContest_scores.median(level=0)
print(bowlingContest_median)
```

```
        score
A_team   88.0
B_team   87.0
C_team   82.5
```

Box plot, also known as box-and-whisker plot and box-and-whisker diagram, is a statistical plot used to show the dispersion of a set of data. The collection of data displayed includes maximum, minimum, median, and upper and lower quartile, so the distribution of data can be seen more clearly using a box plot than a single value. Here it is(from Wikipedia):

```
                            +-----+-+       
  *           o     |-------|   + | |---|
                            +-----+-+    
                                         
+---+---+---+---+---+---+---+---+---+---+   score
0   1   2   3   4   5   6   7   8   9  10
```

This set of data shows minimum=5, the lower quartile(Q1)=7, median(Med, i.e., Q2)=8.5, upper quartile(Q3)=9, maximum=10, average=8, interquartile range=(Q3-Q2)=2(namely ΔQ). Use the plot function provided by Pandas to print the box plot and see the distribution of the scores of each team.

```python
bowlingContest_scores_transpose=bowlingContest_scores.stack().unstack(level=0)
boxplot=bowlingContest_scores_transpose.boxplot(column=['A_team', 'B_team', 'C_team'])
```

<a href=""><img src="./imgs/2_10.png" height="auto" width="auto" title="caDesign"/></a>

 The box diagram provided by Plotly library can interactively display specific values, giving it higher graphical power.

```python
import plotly.express as px
fig = px.box(bowlingContest_scores.xs('C_team',level=0), y="score")
fig.update_layout(
    autosize=False,
    width=500,
    height=500,
    title='C_team'
    )
fig.show()
```

<a href=""><img src="./imgs/2_11.jpg" height="500" width="auto" title="caDesign"/></a>

Find the standard deviation (SD), and the mathematical symbol is often used σ(sigma), used to measure a set of discrete numerical degree, formula:  ：$SD= \sqrt{1/N \sum_{i=1}^N { x_{i}- \mu } } $  , $\mu$ for average. Although A_team and B_team have the same mean value, the distribution of values is very different. By calculating the standard deviation to compare the discrete degree, the smaller the standard deviation is, the smaller the degree of dispersion is.  Conversely, the greater the standard deviation, the greater the degree of dispersion.

```python
bowlingContest_std=bowlingContest_scores.std(level=0)
print(bowlingContest_std)
```

```
            score
A_team  30.172835
B_team  10.373042
C_team  63.033325
```

🐨Back to the experimental data, because the 'detial_info_tag' for the delicacy section of the first-level industry category contains the restaurant sub-category, the box chart shows the sub-category 'detail_info_overall_rating' score's numerical distribution and the calculated standard deviation.

```python
delicacy=poi_gpd.xs('poi_0_delicacy',level=0)
delicacy_rating=delicacy[['detail_info_tag','detail_info_overall_rating']] 
print(delicacy_rating.head())
```

```
  detail_info_tag  detail_info_overall_rating
0          美食;中餐厅                         4.4
2          美食;中餐厅                         4.1
4          美食;中餐厅                         4.5
6          美食;中餐厅                         4.4
8          美食;中餐厅                         4.2
```

Check the restaurant type and remove misclassified data, such as `'教育培训;其他'`. We can also adjust the name of the sub-category, such as by '美食;中餐厅'  changed to '中餐厅' where the `df.applay()` method was used. Finally, it is mapped to English characters to avoid displaying errors when printing. If Chinese characters are displayed incorrectly, corresponding processing statements need to be added.

```python
print(delicacy_rating.detail_info_tag.unique())
delicacy_rating_clean=delicacy_rating[delicacy_rating.detail_info_tag!='教育培训;其他']
print(delicacy_rating_clean.detail_info_tag.unique())
#Defines a function for the `df.apply()` function to handle strings.
def str_row(row):
    if type(row)==str:
        row_=row.split(';')[-1]
    else:
        #print(type(row))
        row_='nan' #The original data type is nan, which, when viewed through `type(row)`, is determined to be a `float` type, converted to a string.
    return row_
delicacy_rating_clean.loc[:,["detail_info_tag"]]=delicacy_rating_clean["detail_info_tag"].apply(str_row)  
print(delicacy_rating_clean.detail_info_tag.unique())

tag_mapping={'中餐厅':'Chinese_restaurant','小吃快餐店':'Snake_bar','nan':'nan','其他':'others','外国餐厅':'Foreign_restaurant','蛋糕甜品店':'CakeANDdessert_shop','咖啡厅':'cafe','茶座':'teahouse','酒吧':'bar'}
delicacy_rating_clean.loc[:,["detail_info_tag"]]=delicacy_rating_clean["detail_info_tag"].replace(tag_mapping)
print(delicacy_rating_clean.detail_info_tag.unique())
```

```
['美食;中餐厅' nan '美食;小吃快餐店' '美食;其他' '美食;外国餐厅' '美食;蛋糕甜品店' '美食;咖啡厅' '美食;茶座'
 '美食;酒吧' '教育培训;其他']
['美食;中餐厅' nan '美食;小吃快餐店' '美食;其他' '美食;外国餐厅' '美食;蛋糕甜品店' '美食;咖啡厅' '美食;茶座'
 '美食;酒吧']
['中餐厅' 'nan' '小吃快餐店' '其他' '外国餐厅' '蛋糕甜品店' '咖啡厅' '茶座' '酒吧']
['Chinese_restaurant' 'nan' 'Snake_bar' 'others' 'Foreign_restaurant'
 'CakeANDdessert_shop' 'cafe' 'teahouse' 'bar']
```

```python
delicacy_rating_clean.boxplot(column=['detail_info_overall_rating'],by=['detail_info_tag'],figsize=(25,8))
delicacy_rating_clean_std=delicacy_rating_clean.set_index(['detail_info_tag']).std(level=0)
print(delicacy_rating_clean_std)
```

```
                     detail_info_overall_rating
detail_info_tag                                
Chinese_restaurant                     0.488348
nan                                         NaN
Snake_bar                              0.490777
others                                 0.367939
Foreign_restaurant                     0.159960
CakeANDdessert_shop                    0.257505
cafe                                   0.243386
teahouse                               0.629374
bar                                    0.189297
```

<a href=""><img src="./imgs/2_12.png" height="auto" width="auto" title="caDesign"/></a>

###### Standard score
🍅A simple data example was established based on the test scores in The Manga Guide to Statistics. In this case, although Mason and Reece had the same score in English and Chinese subjects, and history and biology subjects respectively, their significance was different due to their different standard deviation, i.e., different discrete degree. The smaller the standard deviation, and the lower the degree of dispersion, the change of each unit of value will affect the final ranking. Every point is essential if the standard deviation is small, other students can easily catch up with you, but if the standard deviation is large, other students cannot easily catch up with you. 

```python
test_score_dic={"English":{"Mason":90,"Reece":81,'A':73,'B':97,'C':85,'D':60,'E':74,'F':64,'G':72,'H':67,'I':87,'J':78,'K':85,'L':96,'M':77,'N':100,'O':92,'P':86},
                "Chinese":{"Mason":71,"Reece":90,'A':79,'B':70,'C':67,'D':66,'E':60,'F':83,'G':57,'H':85,'I':93,'J':89,'K':78,'L':74,'M':65,'N':78,'O':53,'P':80},
                "history":{"Mason":73,"Reece":61,'A':74,'B':47,'C':49,'D':87,'E':69,'F':65,'G':36,'H':7,'I':53,'J':100,'K':57,'L':45,'M':56,'N':34,'O':37,'P':70},
                "biology":{"Mason":59,"Reece":73,'A':47,'B':38,'C':63,'D':56,'E':75,'F':53,'G':80,'H':50,'I':41,'J':62,'K':44,'L':26,'M':91,'N':35,'O':53,'P':68},
               }

test_score=pd.DataFrame.from_dict(test_score_dic)
print(test_score.tail())
```

```
       English  Chinese  history  biology
Mason       90       71       73       59
N          100       78       34       35
O           92       53       37       53
P           86       80       70       68
Reece       81       90       61       73
```

The standard score is also known as z-score. Z-score represents the distance between the original value and the mean value. It is calculated in terms of standard deviation, that is, how many standard deviations are there between the point of interest and the mean so that the importance of a particular value can be compared between different sets of data. The formula is: $z=(x- \mu )/ \sigma $, $\sigma \neq 0$  and $x$ is the original score that needs to be standardized, $\mu$ is the average and $\sigma$ is the standard deviation.

Characteristics of standard score

1. No matter how many scores are the full mark of the variable, the mean of the standard score is bound to be 0, and the standard deviation is bound to be 1;
2. Regardless of the unit of variable, the mean value of the standard score must be 0, and the standard deviation is bound to be 1. 

```python
from scipy.stats import zscore
test_Zscore=test_score.apply(zscore)
print(test_Zscore.tail())
```

```
        English   Chinese   history   biology
Mason  0.770054 -0.296174  0.780635  0.162355
N      1.658577  0.325792 -1.083330 -1.298840
O      0.947758 -1.895516 -0.939948 -0.202944
P      0.414644  0.503497  0.637253  0.710303
Reece -0.029617  1.392020  0.207107  1.014719
```

Among them, Mason's standard score in English subject is 0.77, which is 0.77 standard deviation above the average score in the overall distribution. In Chinese, the standard score of Reece is 1.39, which is 1.39 standard deviations above the average in the overall distribution. In other words, the value of each score of Reece is higher than that of Mason, but Mason's score is not easy to be surpassed by others, and also not easy to surpass others.

🐨Back to the experimental data, calculate the standard score for the delicacy section 'detail_info_overall_rating' and  'detail_info_price'.

```python
delicacy=poi_gpd.xs('poi_0_delicacy',level=0)
delicacy_dropna=delicacy.dropna(subset=['detail_info_overall_rating', 'detail_info_price'])
delicacy_Zscore=delicacy_dropna[['detail_info_overall_rating', 'detail_info_price']].apply(zscore).join(delicacy["name"])
print(delicacy_Zscore.head())
```

```
   detail_info_overall_rating  detail_info_price             name
0                    0.180589           1.389133       西安饭庄(锦业路店)
2                   -0.773606           0.505657        范家大院(华为店)
4                    0.498655          -0.015368       苏福记(紫薇臻品店)
6                    0.180589          -0.060674        大厨小馆(绿地店)
8                   -0.455541           0.800149  上下九广州菜馆(锦业路旗舰店)
```

In the experimental data, the importance of z-score in comparing the price and rating of a restaurant can be calculated, which means that the closer the price is to the average, the higher (lower) the corresponding rating is. However, it is difficult to determine whether such a relationship exists by looking at the data of a single restaurant, so we can print out the curve and observe the changing rules of the curve. Because of the large amount of raw data, you need to use the `df.rolling()` method to smooth the data and then plot a curve to observe it. 

```python
cdelicacy_Zscore.rolling(20, win_type='triang').sum().plot.line(figsize=(25,8))
```

<a href=""><img src="./imgs/2_13.png" height="auto" width="auto" title="caDesign"/></a>

As can be observed from the figure above, when the standard score of the price (orange line) is high, the corresponding standard score of the rating usually tends to be lower, and vice versa. If the customized price of the restaurant is close to the average, relatively speaking, the more the evaluation is higher than the average.

### 1.5 key point
#### 1.5.1 data processing technique
* Pandas processing technology summary-A

DataFame and Serie provided by Panda are the most commonly used data formats and are particularly crucial in the geospatial data processing. Pandas provide numerous processing tools. It is unrealistic and undesirable to read through the Pandas handbook to master Pandas. Generally speaking, you can study the introductory manual, get a basic understanding of it, and then search for it when you encounter problems in the process of data processing. It need not see all Pandas functions, but some commonly used functions often can be used, or some functions take us a lot of time to search, or to further improve coding again if you do not sort out,  the next time you meet is unfavorable to even search, you can try to set up your code repository when using it convenient query.

[pandas](https://pandas.pydata.org/) library official document generally give a detailed explanation, including the functions, parameters, attributes, and the actual case, so the Pandas processing technique is simply listed for query purposes, and the specific content can be further retrieved using the search engine with this brief description.

1. data merge

-merge list， `poi_df_concat=pd.concat(poi_df_dic.values(),keys=poi_df_dic.keys(),sort=True)`

-established directly， `ranmenPriceBins_freqANDrelFreq=pd.DataFrame({'fre':ranmenPriceBins_frequency,'relFre':ranmenPriceBins_relativeFrequency})`

-convert Series to DataFrame， `delicacy_price_df=delicacy_price.to_frame(name='price')`

-join DataFrame， `ranmen_fre=ranmenPriceBins_freqANDrelFreq.join(ranmenPriceBins_median).sort_index().reset_index()`

-created from nested dictionaries， `bowlingContest_scores=pd.DataFrame.from_dict(bowlingContest_scores_dic, orient='index').stack().to_frame(name='score')`


2. numerical retrieval

-.loc() mode, with index and column configured

```python
poi_fieldsExtraction=poi_df_concat.loc[:,fields_extraction]
poi_gpd.loc[['poi_0_delicacy'],:].plot(column='detail_info_overall_rating') 
df=poi_gpd.loc[pd.IndexSlice[:,:2],:]
```

-.xs method, multiple index extraction

`delicacy_price=poi_gpd.xs('poi_0_delicacy',level=0).detail_info_price`


3. data manipulation

-apply the function to manipulate the data row by row，

use the anonymous function lambda

`poi_geoDF['geometry']=poi_geoDF.apply(lambda row:Point(row.location_lng,row.location_lat),axis=1)`

use custom function

```python
def str_row(row):
    if type(row)==str:
        row_=row.split(';')[-1]
    else:
        #print(type(row))
        row_='nan' #The original data type is nan, which, when viewed through `type(row)`, is determined to be a `float` type, converted to a string.
    return row_
    
delicacy_rating_clean.loc[:,["detail_info_tag"]]=delicacy_rating_clean["detail_info_tag"].apply(str_row)
```

-shard data according to group spacing，`ranmen_price['price_bins']=pd.cut(x=ranmen_price.price,bins=bins,right=False)`

-sorting rows， `ranmenPrice_bins=ranmen_price.sort_values(by=['price'])` 

-frequency calculation， `ranmenPriceBins_frequency=ranmenPrice_bins.price_bins.value_counts(normalize=True)`

-median calculation， `ranmenPriceBins_median=ranmenPrice_bins.median(level=0)`

-mean calculation， `bowlingContest_mean=bowlingContest_scores.mean(level=0)`

-standard deviation calculation， `bowlingContest_std=bowlingContest_scores.std(level=0)`

-standard score calculation， 

```python
from scipy.stats import zscore
test_Zscore=test_score.apply(zscore)
```

-remove empty value， `delicacy_dropna=delicacy.dropna(subset=['detail_info_overall_rating', 'detail_info_price'])`

-smooth data， `delicacy_Zscore.rolling(20, win_type='triang').sum().plot.line(figsize=(25,8))`

4. the index operation（(multi)index and columns）

-reset the index， `df=df.reset_index()`

-set (multiple) indexes， `ranmenPrice_bins.set_index(['price_bins',ranmenPrice_bins.index],drop=False,inplace=True)`

-multiple index sharding， `df=poi_gpd.loc[pd.IndexSlice[:,:2],:]`

-rename， `ranmenPriceBins_median.rename(columns={'price':'median'},inplace=True)`

-organize the structure， `bowlingContest_scores_transpose=bowlingContest_scores.stack().unstack(level=0)`

5. chart plot

-histogram，`ranmen_fre.loc[:,['fre','index']].plot.bar(x='index',rot=0,figsize=(5,5))`

-boxplot，`boxplot=bowlingContest_scores_transpose.boxplot(column=['A_team', 'B_team', 'C_team'])`

6. other

-copy and deep copy `poi_geoDF=poi_fieldsExtraction.copy(deep=True)`

* GeoPandas data saving types include： Shapefile(.shp)，GeoJSON(.geojson)，GeoPackage(.gpkg) and PostGIS，Examples are listed below (from GeoPandas handbook)：

```python
countries_gdf.to_file("countries.shp")
countries_gdf.to_file("countries.geojson", driver='GeoJSON')

countries_gdf.to_file("package.gpkg", layer='countries', driver="GPKG")
cities_gdf.to_file("package.gpkg", layer='cities', driver="GPKG")

from sqlalchemy import create_engine
db_connection_url = "postgres://myusername:mypassword@myhost:5432/mydatabase";
engine = create_engine(db_connection_url)
countries_gdf.to_postgis(name="countries_table", con=engine)
```

#### 1.5.2 The newly created function tool
* function-Baidu Map open platform POI data mass crawling，`baiduPOI_batchCrawler(poi_config_para)`。need to call the single POI classification crawler function. baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False)
* funciton-With the folder path as the key, the value is a list of all filenames under the folder. File types can be specified.  `filePath_extraction(dirpath,fileType)`
* funciton-.csv format  POI data is converted to GeoDataFrame in batch，`poi_csv2GeoDF_batch(poi_paths,fields_extraction,save_path)`。 which needs to call .csv to DataFrame data format provided by pandas function `csv2df(poi_fn_csv)`.
* funciton-use Plotly to display data in DataFrame format as a table,`ployly_table(df,column_extraction)`
* function-frequency calculation，frequency_bins(df,bins)

#### 1.5.3 The python libraries that are being imported

```python
import util_poi
import os, pathlib
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
```