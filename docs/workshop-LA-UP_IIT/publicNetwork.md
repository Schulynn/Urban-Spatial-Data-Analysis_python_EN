> Created on Mon Mar  1 10\31\37 2021 @author: richie bao -workshop-LA-UP_IIT

## Composite network architecture: target accessibility of public transport system


### 1.1 The establishment of the traffic network structure of the bus system
#### 1.1.1 Read the required data.

In the data preprocessing stage, the data that may be needed has been written into the PostgreSQL database.The database reading function is defined below to facilitate the reading(as GeoDataFrame format) of arbitrary data from the database.


```python
def postSQL2gpd(table_name,geom_col='geometry',**kwargs):
    from sqlalchemy import create_engine
    import geopandas as gpd
    
    engine=create_engine("postgres://{myusername}:{mypassword}@localhost:5432/{mydatabase}".format(myusername=kwargs['myusername'],mypassword=kwargs['mypassword'],mydatabase=kwargs['mydatabase']))  
    gdf=gpd.read_postgis(table_name, con=engine,geom_col=geom_col)
    print("_"*50)
    print('The data has been read from PostSQL database...')    
    return gdf
```

* Read the geometric center point of the analysis site boundary used to extract the adjacent bus stops at the specified distance.


```python
b_centroid=postSQL2gpd(table_name='b_centroid',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
b_centroid
```

    __________________________________________________
    The data has been read from PostSQL database...
    

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\sqlalchemy\dialects\postgresql\base.py:3199: SAWarning: Did not recognize type 'geometry' of column 'geometry'
      "Did not recognize type '%s' of column '%s'" % (attype, name)
    




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
      <th>x</th>
      <th>y</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>665745.498479</td>
      <td>3.547575e+06</td>
      <td>POINT (665745.498 3547575.227)</td>
    </tr>
  </tbody>
</table>
</div>




```python
b_centroid.plot() #figsize=(20,10)
```




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_01.png" height='auto' width='auto' title="caDesign"></a> 
    


* Read bus stop and route data.


```python
bus_stations=postSQL2gpd(table_name='bus_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
bus_routes=postSQL2gpd(table_name='bus_routes',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\sqlalchemy\dialects\postgresql\base.py:3199: SAWarning: Did not recognize type 'geometry' of column 'geometry'
      "Did not recognize type '%s' of column '%s'" % (attype, name)
    

    __________________________________________________
    The data has been read from PostSQL database...
    __________________________________________________
    The data has been read from PostSQL database...
    


```python
import matplotlib.pyplot as plt
fig, ax=plt.subplots(figsize=(100,10))
bus_routes.plot(ax=ax)
bus_stations.plot(ax=ax,marker='o',color='red',markersize=50)
plt.show()
```


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_02.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
plt.rcParams['font.sans-serif']=['SimHei']
ax=bus_stations.plot(color='red',markersize=5,figsize=(30,30))
for x,y,label in zip(bus_stations.geometry.x,bus_stations.geometry.y,bus_stations.PointName):
    ax.annotate(label,xy=(x,y),xytext=(3,3),textcoords="offset points",fontsize=10)
```


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_03.png" height='auto' width='auto' title="caDesign"></a> 
    


#### 1.1.2 Establish the bus network structure.
Search for bus stops (`start_stops_PointUid`) within a given 500 distance from the site's geometric center for potential arrival stops on the site. 

Establish a bus network based on bus stops and bus routes. The network is an undirected network structure between two sites back and forth, rather than one-way. Because of the bus and subway and other possible transport composite networks, different vehicles have different speeds and are unified as time travel cost rather than distances.


```python
def bus_network(b_centroid_,bus_stations_,bus_routes_,speed,**kwargs): #
    import copy
    import pandas as pd
    import networkx as nx
    from shapely.ops import nearest_points
    from shapely.ops import substring
    from tqdm import tqdm
    
    #compute the distance between the site centroid and each bus station and get the nearest ones by given threshold
    bus_stations=copy.deepcopy(bus_stations_)
    bus_stations['center_distance']=bus_stations.geometry.apply(lambda row:row.distance(b_centroid_.geometry.values[0]))
    # bus_stations.sort_values(by=['center_distance'],inplace=True)
    # print(bus_stations)
    start_stops=bus_stations[bus_stations.center_distance<=kwargs['start_stops_distance']]
    start_stops_lineUID=start_stops.LineUid.unique()   
    start_stops_PointUid=start_stops.PointUid.unique()   
    
    #build bus stations network
    bus_staions_routes=pd.merge(bus_stations,bus_routes_,on='LineUid')
    bus_staions_routes_idx_LineUid=bus_staions_routes.set_index('LineUid',append=True,drop=False)    
    
    lines_group_list=[]
    s_e_nodes=[]
    # i=0
    for LineUid,sub_df in tqdm(bus_staions_routes_idx_LineUid.groupby(level=1)):
        # print(sub_df)
        # print(sub_df.columns)
        sub_df['nearestPts']=sub_df.apply(lambda row:nearest_points(row.geometry_y,row.geometry_x)[0],axis=1)
        sub_df['project_norm']=sub_df.apply(lambda row:row.geometry_y.project(row.nearestPts,normalized=True),axis=1)
        sub_df.sort_values(by='project_norm',inplace=True)
        sub_df['order_idx']=range(1,len(sub_df)+1)
        # station_geometries=sub_df.geometry_x.to_list()
        project=sub_df.project_norm.to_list()
        sub_df['second_project']=project[1:]+project[:1]
        
        PointName=sub_df.PointName.to_list()
        sub_df['second_PointName']=PointName[1:]+PointName[:1]
        PointUid=sub_df.PointUid.to_list()
        sub_df['second_PointUid']= PointUid[1:]+ PointUid[:1]
        
        sub_df['substring']=sub_df.apply(lambda row:substring(row.geometry_y,row.project_norm,row.second_project,normalized=True),axis=1)
        sub_df['forward_length']=sub_df.apply(lambda row:row.substring.length,axis=1)
        sub_df['time_cost']=sub_df.apply(lambda row:row.forward_length/(speed*1000)*60,axis=1)
        
        sub_df['edges']=sub_df.apply(lambda row:[(row.PointUid,row.second_PointUid),(row.second_PointUid,row.PointUid)],axis=1)
        
        lines_group_list.append(sub_df)
        s_e_nodes.append(sub_df.edges.to_list()[-1][0])
        
        # print(i)
        # i+=1
    lines_df4G=pd.concat(lines_group_list)
    
    # G=nx.Graph()
    G=nx.from_pandas_edgelist(df=lines_df4G,source='PointUid',target='second_PointUid',edge_attr=['PointName','second_PointName','forward_length','geometry_x','time_cost'])
    for idx,row in lines_df4G.iterrows():
        G.nodes[row['PointUid']]['position']=(row.geometry_x.x,row.geometry_x.y)
        G.nodes[row['PointUid']]['station_name']=row.PointName        
    
    return G,s_e_nodes,start_stops_PointUid,lines_df4G
```


```python
G_bus_stations,s_e_nodes,start_stops_PointUid,lines_df4G=bus_network(b_centroid,bus_stations,bus_routes,speed=45,start_stops_distance=500) #bus_network_structure
```

    100%|██████████| 334/334 [00:04<00:00, 67.87it/s]
    

Remove the leading and trailing edges of the path in the network structure. That is the connection between the first station and the last station.


```python
G_bus_stations.remove_edges_from(s_e_nodes)
```

The bus stops are adjacent to the site within a given distance.


```python
print(start_stops_PointUid.shape)
start_stops_PointUid_unique=np.unique(start_stops_PointUid)
print('After removing potentially duplicated sites,sites number:',start_stops_PointUid_unique.shape)
start_stops_gdf=bus_stations[bus_stations.PointUid.isin(start_stops_PointUid_unique)]
start_stops_gdf.plot()
```

    (78,)
    After removing potentially duplicated sites,sites number: (78,)
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_04.png" height='auto' width='auto' title="caDesign"></a> 
    


* Define a function to draw the network to view the network structure.


```python
import networkx as nx
import numpy as np

def G_draw(G,layout='spring_layout',node_color=None,node_size=None,figsize=(30, 30),font_size=12,edge_color=None,labels=None,with_labels=False):    
    import matplotlib
    import matplotlib.pyplot as plt
    import networkx as nx
    '''
    function - To show a networkx graph
    '''
    #解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['DengXian'] # 指定默认字体 'KaiTi','SimHei'
    plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
    fig, ax=plt.subplots(figsize=figsize)
    #nx.draw_shell(G, with_labels=True)
    layout_dic={
        'spring_layout':nx.spring_layout,   
        'random_layout':nx.random_layout,
        'circular_layout':nx.circular_layout,
        'kamada_kawai_layout':nx.kamada_kawai_layout,
        'shell_layout':nx.shell_layout,
        'spiral_layout':nx.spiral_layout,
    }

    nx.draw(G,nx.get_node_attributes(G,'position'),with_labels=with_labels,labels=labels,node_color=node_color,node_size=node_size,font_size=font_size,edge_color=edge_color,ax=ax)  #nx.draw(G, pos, font_size=16, with_labels=False)
    return ax
```

The red dots are the stops where the site is adjacent. The color of the edge is the time cost as the weight value.


```python
ax=G_draw(G_bus_stations,edge_color=list(nx.get_edge_attributes(G_bus_stations, 'time_cost').values()),labels=nx.get_node_attributes(G_bus_stations,'station_name'),font_size=20,figsize=(20, 20))
G_start_stops=G_bus_stations.subgraph(start_stops_PointUid_unique)
pos=nx.get_node_attributes(G_start_stops,'position')
nx.draw_networkx_nodes(G_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
```




    <matplotlib.collections.PathCollection at 0x22c54246d88>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_05.png" height='auto' width='auto' title="caDesign"></a> 
    


The bus network structure needs to consider transfers. During the transfer process, it may be necessary to walk some distance to another stop, so given a 200m, search for the next bus stops. If the bus's speed is still used, the cost of walking distance is calculated by providing a ratio, or the time cost is calculated directly.


```python
def transfer_stations_network(station_geometries_df,transfer_distance,transfer_weight_ratio,speed): 
    import copy
    from tqdm import tqdm
    import pandas as pd
    import networkx as nx
    
    transfer_df_list=[]
    station_geometries_dict=station_geometries_df.to_dict('record')
    #i=0
    for pt in tqdm(station_geometries_dict):
        station_geometries_df_=copy.deepcopy(station_geometries_df)
        station_geometries_df_['distance']=station_geometries_df_.geometry_x.apply(lambda row:row.distance(pt['geometry_x']))
        
        transfer_df=station_geometries_df_[station_geometries_df_.distance<=transfer_distance]
        transfer_df=transfer_df[transfer_df.distance!=0]
        transfer_df.drop_duplicates(subset='PointUid',keep='first',inplace=True)        
        
        transfer_df['source_station']=pt['PointUid']
        transfer_df['forward_length']=transfer_df.distance*transfer_weight_ratio
        # print(transfer_df['forward_length'])
        transfer_df=transfer_df[transfer_df.LineUid!=pt['LineUid']]      
        # print(transfer_df)    
  
        transfer_df_list.append(transfer_df)
        
        #if i==1:break
        #i+=1
       
    transfer_df_concat=pd.concat(transfer_df_list)
    transfer_df_concat['time_cost']=transfer_df_concat.apply(lambda row:row.forward_length/(speed*1000)*60,axis=1)
    # print(transfer_df_concat)
    G=nx.from_pandas_edgelist(df=transfer_df_concat,source='source_station',target='PointUid',edge_attr=['forward_length','time_cost'])
    
    for idx,row in transfer_df_concat.iterrows():
        G.nodes[row['PointUid']]['position']=(row.geometry_x.x,row.geometry_x.y)
    
    return  G,transfer_df_concat
```


```python
station_geometries_df=lines_df4G[['PointUid','geometry_x','LineUid']] 
G_bus_transfer,transfer_df_concat=transfer_stations_network(station_geometries_df,transfer_distance=200,transfer_weight_ratio=9,speed=45) 
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\pandas\core\frame.py:1554: FutureWarning: Using short name for 'orient' is deprecated. Only the options: ('dict', list, 'series', 'split', 'records', 'index') will be used in a future version. Use one of the above to silence this warning.
      FutureWarning,
    100%|██████████| 4132/4132 [01:13<00:00, 56.47it/s]
    

Merge the bus network and the transfer network. The red lines in the figure are the transfer network.


```python
G_bus=nx.compose(G_bus_stations,G_bus_transfer)
edge_color_=list(nx.get_edge_attributes(G_bus, 'time_cost').values()) #'forward_length'
# edge_color=[i/max(edge_color_) for i in edge_color_]
ax=G_draw(G_bus,edge_color=edge_color_,labels=nx.get_node_attributes(G_bus_stations,'station_name'),font_size=20,figsize=(20, 20))
nx.draw_networkx_nodes(G_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
pos_edge=nx.get_node_attributes(G_bus_transfer,'position')
nx.draw_networkx_edges(G_bus_transfer,pos=pos_edge,edgelist=G_bus_transfer.edges,edge_color='red',ax=ax)
```




    <matplotlib.collections.LineCollection at 0x22c54428b48>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_06.png" height='auto' width='auto' title="caDesign"></a> 
    


#### 1.1.3 Calculate the shortest distance between the stops adjacent to the site and all sites.


```python
def bus_shortest_paths(G,start_stops_PointUid):
    from tqdm import tqdm
    
    all_shortest_length_dict={}
    all_shortest_path_dict={}
    start_stops_PointUi_list=start_stops_PointUid.tolist()
    for stop in tqdm(start_stops_PointUi_list):
        shortest_path=nx.shortest_path(G, source=stop,weight="time_cost")
        shortest_length=nx.shortest_path_length(G, source=stop,weight="time_cost")
        all_shortest_length_dict[stop]=shortest_length
        all_shortest_path_dict[stop]=shortest_path
        # break

    return all_shortest_path_dict,all_shortest_length_dict
```


```python
all_shortest_path_dict,all_shortest_length_dict=bus_shortest_paths(G_bus,start_stops_PointUid)
```

    100%|██████████| 78/78 [00:06<00:00, 12.37it/s]
    

Feel free to print to see one of the shortest paths.


```python
shortest_paths_list=list(all_shortest_path_dict.values())[10]
shortest_path=list(shortest_paths_list.values())[3000]
H=G_bus.subgraph(shortest_path)
ax=G_draw(H,edge_color=list(nx.get_edge_attributes(H, 'time_cost').values()),labels=nx.get_node_attributes(H,'station_name'),font_size=20,figsize=(10, 10),with_labels=True)
nx.draw_networkx_nodes(G_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
```




    <matplotlib.collections.PathCollection at 0x22c19205208>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_07.png" height='auto' width='auto' title="caDesign"></a> 
    


#### 1.1.4 Regional accessibility indicators

The shortest path is calculated by taking the time cost as the weight is divided into 5,10, and 15-time distances. The number of shortest paths can get obtained for each segment. The more the number, the more chances you have to get there. According to the calculated results, the mean values are 1405 when less than 5 minutes, 2336 when less than 10 minutes, greater than or equal to 5 minutes, and 31 when more than 10 minutes. The preliminary conclusion can be drawn as follows: as for the accessibility of the analyzed site, the accessibility of the time distance greater than 10 minutes is significantly reduced under the bus traffic network.


```python
def bus_service_index(G,all_shortest_path_dict,all_shortest_length_dict,**kwargs):
    import pandas as pd
    from tqdm import tqdm
    
    start,end,step=kwargs['start'],kwargs['end'],kwargs['step']      #start,end,step=0,10000,2000;0,25,5
    def partition(value,start,end,step):
        import numpy as np
        ranges_list=list(range(start,end+step,step))
        ranges=[(ranges_list[i],ranges_list[i+1]) for i in  range(len(ranges_list)-1)]

        for r in ranges:
            # print(r,value)
            if r[0]<=value<r[1] :    
                return r[1]
    
    shortest_df_dict={}
    shortest_routes_fre={}
    for start_stop,shortest_length in tqdm(all_shortest_length_dict.items()):
        # print(start_stop)
        shortest_df=pd.DataFrame.from_dict(shortest_length, orient='index',columns=['duration'])  
        shortest_df['path']=shortest_df.index.map(all_shortest_path_dict[start_stop])
        shortest_df['range']=shortest_df.duration.apply(partition,args=(start,end,step))
        # print(shortest_df.range)
        ranges_frequency=shortest_df['range'].value_counts()
        shortest_routes_fre[start_stop]=ranges_frequency.to_dict()
        # print(ranges_frequency.to_dict())
        shortest_df_dict[start_stop]=shortest_df

    shortest_routes_fre_df=pd.DataFrame.from_dict(shortest_routes_fre,orient='index')
    
    return shortest_routes_fre_df,shortest_df_dict
```


```python
shortest_routes_fre_df,shortest_df_dict=bus_service_index(G_bus,all_shortest_path_dict,all_shortest_length_dict,start=0,end=25,step=5)   
shortest_routes_fre_df_statistics=shortest_routes_fre_df.describe(include='all')
shortest_routes_fre_df_statistics
```

    100%|██████████| 78/78 [00:01<00:00, 73.78it/s]
    




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
      <th>10</th>
      <th>5</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>78.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2740.602564</td>
      <td>1366.602564</td>
      <td>20.794872</td>
    </tr>
    <tr>
      <th>std</th>
      <td>96.377416</td>
      <td>101.027621</td>
      <td>10.044523</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2632.000000</td>
      <td>1178.000000</td>
      <td>8.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2652.000000</td>
      <td>1255.500000</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2679.000000</td>
      <td>1428.000000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2850.000000</td>
      <td>1460.000000</td>
      <td>23.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2902.000000</td>
      <td>1478.000000</td>
      <td>48.000000</td>
    </tr>
  </tbody>
</table>
</div>



The time step value can be reduced to 1 minute, and the variation trend of the number of shortest paths can be observed with each minute change. Six minutes is the top trend.


```python
shortest_routes_fre_df_,shortest_df_dict_=bus_service_index(G_bus,all_shortest_path_dict,all_shortest_length_dict,start=0,end=25,step=1)   
shortest_routes_fre_df_statistics_=shortest_routes_fre_df_.describe(include='all')
shortest_routes_fre_df_statistics_
```

    100%|██████████| 78/78 [00:01<00:00, 43.42it/s]
    




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
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>5</th>
      <th>9</th>
      <th>4</th>
      <th>10</th>
      <th>3</th>
      <th>2</th>
      <th>1</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>78.00000</td>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>78.000000</td>
      <td>65.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>802.371795</td>
      <td>754.961538</td>
      <td>675.410256</td>
      <td>585.551282</td>
      <td>402.679487</td>
      <td>359.50000</td>
      <td>105.179487</td>
      <td>238.884615</td>
      <td>137.858974</td>
      <td>44.807692</td>
      <td>18.192308</td>
      <td>3.123077</td>
    </tr>
    <tr>
      <th>std</th>
      <td>36.820064</td>
      <td>59.952241</td>
      <td>28.470735</td>
      <td>25.787668</td>
      <td>97.171490</td>
      <td>45.79677</td>
      <td>50.892488</td>
      <td>32.271970</td>
      <td>16.922838</td>
      <td>14.099883</td>
      <td>8.702974</td>
      <td>2.080472</td>
    </tr>
    <tr>
      <th>min</th>
      <td>706.000000</td>
      <td>638.000000</td>
      <td>624.000000</td>
      <td>528.000000</td>
      <td>289.000000</td>
      <td>266.00000</td>
      <td>46.000000</td>
      <td>179.000000</td>
      <td>96.000000</td>
      <td>14.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>790.000000</td>
      <td>734.000000</td>
      <td>652.250000</td>
      <td>568.500000</td>
      <td>314.000000</td>
      <td>329.00000</td>
      <td>61.500000</td>
      <td>210.250000</td>
      <td>122.000000</td>
      <td>38.750000</td>
      <td>12.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>803.500000</td>
      <td>748.000000</td>
      <td>667.500000</td>
      <td>585.000000</td>
      <td>358.000000</td>
      <td>368.50000</td>
      <td>78.000000</td>
      <td>246.500000</td>
      <td>137.000000</td>
      <td>50.000000</td>
      <td>15.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>828.500000</td>
      <td>781.750000</td>
      <td>705.000000</td>
      <td>605.000000</td>
      <td>489.750000</td>
      <td>388.75000</td>
      <td>144.750000</td>
      <td>271.000000</td>
      <td>153.750000</td>
      <td>54.750000</td>
      <td>21.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>864.000000</td>
      <td>904.000000</td>
      <td>720.000000</td>
      <td>633.000000</td>
      <td>565.000000</td>
      <td>423.00000</td>
      <td>218.000000</td>
      <td>284.000000</td>
      <td>162.000000</td>
      <td>61.000000</td>
      <td>46.000000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
shortest_routes_fre_df_statistics_.T.sort_index().plot(y='mean')
plt.xticks(shortest_routes_fre_df_statistics_.T.index)
plt.show()
```


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_08.png" height='auto' width='auto' title="caDesign"></a> 
    


#### 1.1.5 Covering area with the shortest distance in different intervals


```python
flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]    

def G_draw_path(G,paths,figsize=(30, 30),node_size=200):
    import networkx as nx
    import matplotlib.pyplot as plt
    import random
    fig, ax=plt.subplots(figsize=figsize)
    
    G_p=G.copy()
    G_p.remove_edges_from(list(G_p.edges()))
    G_p_range=nx.Graph(G_p.subgraph(flatten_lst(paths)))
    
    edges=[]
    for r in paths:
        path_edges=[(r[n],r[n+1]) for n in range(len(r)-1)]
        G_p_range.add_edges_from(path_edges)
        edges.append(path_edges)
    print("Graph has %d nodes with %d paths(edges)" %(G_p_range.number_of_nodes(), G_p_range.number_of_edges()))   
    pos=nx.get_node_attributes(G_p_range,'position')
    nx.draw_networkx_nodes(G_p_range,pos=pos,node_size=node_size) #,with_labels=True,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=20
    # nx.draw_networkx_labels(G_p_range,pos=pos,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=8)
    # colors = ['r', 'b', 'y']
    # linewidths = [20,10,5]    
    for ctr, edgelist in enumerate(edges):
        color=(random.random(), random.random(),random.random())
        nx.draw_networkx_edges(G_p_range,pos=pos,edgelist=edgelist,edge_color=color,node_size=node_size) #edge_color = colors[ctr], width=linewidths[ctr]
    # plt.savefig('G_p_range.png')

def G_draw_range_paths(G,range_paths,figsize=(30, 30),node_size=200):
   
    ranges=range_paths.range.unique()
    for i in ranges:
        range_paths_range=range_paths[range_paths.range==i]
        path_edges=range_paths_range.path.to_list()
        G_draw_path(G,path_edges,figsize=figsize,node_size=node_size)

def G_draw_paths_composite(G,range_paths,figsize=(30, 30),node_size=200):
    import networkx as nx
    import matplotlib.pyplot as plt
    import random
    from collections import defaultdict
    
    fig, ax=plt.subplots(figsize=figsize)
    
    ranges=range_paths.range.unique()
    print(ranges)
    path_edges_dict={}
    for i in ranges:
        range_paths_range=range_paths[range_paths.range==i]
        path_edges=range_paths_range.path.to_list()    
        path_edges_dict[i]=path_edges
    
    G_p=G.copy()
    G_p.remove_edges_from(list(G_p.edges()))
    G_p_range=nx.Graph(G_p.subgraph(flatten_lst(list(path_edges_dict.values()))))
    
    edges_dict=defaultdict(list)
    for k in path_edges_dict.keys():
        for r in path_edges_dict[k]:
            path_edges=[(r[n],r[n+1]) for n in range(len(r)-1)]
            G_p_range.add_edges_from(path_edges)
            edges_dict[k].append(path_edges)
    print("Graph has %d nodes with %d paths(edges)" %(G_p_range.number_of_nodes(), G_p_range.number_of_edges()))   
        
    pos=nx.get_node_attributes(G_p_range,'position')
    nx.draw_networkx_nodes(G_p_range,pos=pos,node_size=node_size) #,with_labels=True,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=20
    #nx.draw_networkx_labels(G_p_range,pos=pos,labels=nx.get_node_attributes(G_p_range,'station_name'),font_size=8)
    linewidths=[1,5,9,] #[5,10,15,20,25
    linewidths.reverse()
    colors=['red','blue','green']
    for j,k in enumerate(reversed(list(edges_dict.keys()))):
        # color=(random.random(), random.random(),random.random())
        for i, edgelist in enumerate(edges_dict[k]):            
            nx.draw_networkx_edges(G_p_range,pos=pos,edgelist=edgelist,edge_color=colors[j],width=linewidths[j]) #edge_color = colors[ctr], width=linewidths[ctr]
    # plt.savefig('G_p_range.png')    
    
    return edges_dict,ax
```


```python
paths_0=list(shortest_df_dict.values())[0]
G_draw_range_paths(G_bus,paths_0,figsize=(10, 10),node_size=10)
```

    Graph has 1215 nodes with 1214 paths(edges)
    Graph has 3408 nodes with 3407 paths(edges)
    Graph has 244 nodes with 243 paths(edges)
    


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_09.png" height='auto' width='auto' title="caDesign"></a> 
    



    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_10.png" height='auto' width='auto' title="caDesign"></a> 
    



    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_11.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
edges_dict,ax=G_draw_paths_composite(G_bus,paths_0,figsize=(15,15),node_size=10)
nx.draw_networkx_nodes(G_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
```

    [ 5 10 15]
    Graph has 4128 nodes with 4127 paths(edges)
    




    <matplotlib.collections.PathCollection at 0x22c45b43108>




<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_12.png" height='auto' width='auto' title="caDesign"></a> 
    


* Write the network to disk.


```python
nx.write_gpickle(G_bus, "./results/G_bus.gpickle")
```

### 1.2 The establishment of the traffic network structure of the subway system, and the composite network of buses and subways.

The establishment method of the subway traffic network is the same as that of the bus traffic network.

#### 1.2.1 Building a subway network


```python
subway_stations=postSQL2gpd(table_name='subway_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
subway_routes=postSQL2gpd(table_name='subway_lines',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')

G_subway_stations,subway_s_e_nodes,subway_start_stops_PointUid,subway_lines_df4G=bus_network(b_centroid,subway_stations,subway_routes,speed=60*2,start_stops_distance=1200) #bus_network_structure
G_subway_stations.remove_edges_from(subway_s_e_nodes)
G_draw(G_subway_stations,edge_color=list(nx.get_edge_attributes(G_subway_stations, 'time_cost').values()),labels=nx.get_node_attributes(G_subway_stations,'station_name'),font_size=20,figsize=(10, 10),with_labels=True)
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\sqlalchemy\dialects\postgresql\base.py:3199: SAWarning: Did not recognize type 'geometry' of column 'geometry'
      "Did not recognize type '%s' of column '%s'" % (attype, name)
    100%|██████████| 8/8 [00:00<00:00, 95.49it/s]

    __________________________________________________
    The data has been read from PostSQL database...
    __________________________________________________
    The data has been read from PostSQL database...
    

    
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_13.png" height='auto' width='auto' title="caDesign"></a> 
    


#### 1.2.2 Establish a network between subway stations and bus stations


```python
def transfer_stations_network_subway_bus(bus_df,subway_df,transfer_distance,transfer_weight_ratio,speed): 
    import copy
    from tqdm import tqdm
    import pandas as pd
    import networkx as nx
    
    transfer_df_list=[]
    bus_stations_=bus_df[['PointUid','geometry_x','LineUid']]
    subway_station_=subway_df[['PointUid','geometry_x','LineUid']]
    subway_station=subway_station_.to_dict('recorde')
    for pt in tqdm(subway_station):
        # print(pt)
        bus_stations=copy.deepcopy(bus_stations_)
        bus_stations['distance']=bus_stations.geometry_x.apply(lambda row:row.distance(pt['geometry_x']))
        # print(bus_stations)
        transfer_df=bus_stations[bus_stations.distance<=transfer_distance]
        transfer_df=transfer_df[transfer_df.distance!=0]
        transfer_df.drop_duplicates(subset='PointUid',keep='first',inplace=True)   
        # print(transfer_df)
        
        transfer_df['source_station']=pt['PointUid']
        transfer_df['forward_length']=transfer_df.distance*transfer_weight_ratio
        transfer_df_list.append(transfer_df)
   
        # break
    transfer_df_concat=pd.concat(transfer_df_list)
    transfer_df_concat['time_cost']=transfer_df_concat.apply(lambda row:row.forward_length/(speed*1000)*60,axis=1)
    G=nx.from_pandas_edgelist(df=transfer_df_concat,source='source_station',target='PointUid',edge_attr=['forward_length','time_cost'])
    
    
    return G,transfer_df_concat
```


```python
G_subway_bus_transfer,subway_bus_transfer_df_concat=transfer_stations_network_subway_bus(bus_df=lines_df4G,subway_df=subway_lines_df4G,transfer_distance=300,transfer_weight_ratio=9,speed=45)   #2 
G_subway_bus=nx.compose(G_bus,G_subway_stations)
G_subway_bus_transfer=nx.compose(G_subway_bus,G_subway_bus_transfer)
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\pandas\core\frame.py:1554: FutureWarning: Using short name for 'orient' is deprecated. Only the options: ('dict', list, 'series', 'split', 'records', 'index') will be used in a future version. Use one of the above to silence this warning.
      FutureWarning,
    100%|██████████| 56/56 [00:00<00:00, 60.70it/s]
    

#### 1.2.3 The shortest path of a composite network of subway and bus stations

Because the subway lines are very few compared with the bus lines, it plays a small role in the whole network. Because the subway usually arrives and leaves on time, the subway is usually the first choice, so the subway's time cost is reduced to improve the accuracy of the complex network expression.


```python
all_shortest_path_dict_,all_shortest_length_dict_=bus_shortest_paths(G_subway_bus_transfer,start_stops_PointUid)
shortest_routes_fre_df_,shortest_df_dict_=bus_service_index(G_subway_bus_transfer,all_shortest_path_dict_,all_shortest_length_dict_,start=0,end=25,step=5) 
```

    100%|██████████| 78/78 [00:06<00:00, 11.39it/s]
    100%|██████████| 78/78 [00:01<00:00, 72.49it/s]
    

From the figure, we can observe the shortest path distribution regions of different time cost distances to preliminarily judge which areas are more comfortable to reach the target point and which areas need more time.


```python
paths_0_=list(shortest_df_dict_.values())[0]
edges_dict_,ax=G_draw_paths_composite(G_subway_bus_transfer,paths_0_,figsize=(15,15),node_size=10)
nx.draw_networkx_nodes(G_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
```

    [ 5 10 15]
    Graph has 4162 nodes with 4161 paths(edges)
    




    <matplotlib.collections.PathCollection at 0x22c2bb69cc8>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_14.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
nx.write_gpickle(G_subway_bus_transfer, "./results/G_subway_bus_transfer.gpickle") #Write the network to disk.
```

### 1.3  Comparison of different target sites

Select a new site for analysis, such as Phoenix Hill, and use the same analysis process to compare the differences between the two public transport network structures.

#### 1.3.1 Computation part
Call the above function directly


```python
phoenix_b_centroid=postSQL2gpd(table_name='phoenix_b_centroid',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
phoenix_b_centroid
```

    __________________________________________________
    The data has been read from PostSQL database...
    

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\sqlalchemy\dialects\postgresql\base.py:3199: SAWarning: Did not recognize type 'geometry' of column 'geometry'
      "Did not recognize type '%s' of column '%s'" % (attype, name)
    




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
      <th>x</th>
      <th>y</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>652644.713976</td>
      <td>3.548719e+06</td>
      <td>POINT (652644.714 3548719.132)</td>
    </tr>
  </tbody>
</table>
</div>




```python
phoenix_b_centroid.plot()
```




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_15.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
phoenix_bus_stations=postSQL2gpd(table_name='phoenix_bus_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
phoenix_bus_routes=postSQL2gpd(table_name='phoenix_bus_routes',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\sqlalchemy\dialects\postgresql\base.py:3199: SAWarning: Did not recognize type 'geometry' of column 'geometry'
      "Did not recognize type '%s' of column '%s'" % (attype, name)
    

    __________________________________________________
    The data has been read from PostSQL database...
    __________________________________________________
    The data has been read from PostSQL database...
    


```python
import matplotlib.pyplot as plt
fig, ax=plt.subplots(figsize=(100,10))
phoenix_bus_routes.plot(ax=ax)
phoenix_bus_stations.plot(ax=ax,marker='o',color='red',markersize=50)
plt.show()
```


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_16.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
plt.rcParams['font.sans-serif']=['SimHei']
ax=phoenix_bus_stations.plot(color='red',markersize=5,figsize=(30,30))
for x,y,label in zip(phoenix_bus_stations.geometry.x,phoenix_bus_stations.geometry.y,phoenix_bus_stations.PointName):
    ax.annotate(label,xy=(x,y),xytext=(3,3),textcoords="offset points",fontsize=10)
```


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_17.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
G_phoenix_bus_stations,phoenix_s_e_nodes,phoenix_start_stops_PointUid,phoenix_lines_df4G=bus_network_(phoenix_b_centroid,phoenix_bus_stations,phoenix_bus_routes,speed=45,start_stops_distance=500) #bus_network_structure
len(G_phoenix_bus_stations.edges)
```

    100%|██████████| 75/75 [00:01<00:00, 71.20it/s]
    




    831




```python
G_phoenix_bus_stations.remove_edges_from(phoenix_s_e_nodes)
len(G_phoenix_bus_stations.edges)
```




    756




```python
print(phoenix_start_stops_PointUid.shape)
phoenix_start_stops_PointUid_unique=np.unique(phoenix_start_stops_PointUid)
print('After removing potentially duplicated sites,sites number:',phoenix_start_stops_PointUid_unique.shape)
phoenix_start_stops_gdf=phoenix_bus_stations[phoenix_bus_stations.PointUid.isin(phoenix_start_stops_PointUid_unique)]
phoenix_start_stops_gdf.plot()
```

    (45,)
    After removing potentially duplicated sites,sites number: (45,)
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_18.png" height='auto' width='auto' title="caDesign"></a> 



```python
ax=G_draw(G_phoenix_bus_stations,edge_color=list(nx.get_edge_attributes(G_phoenix_bus_stations, 'time_cost').values()),labels=nx.get_node_attributes(G_phoenix_bus_stations,'station_name'),font_size=20,figsize=(20, 20))
G_phoenix_start_stops=G_phoenix_bus_stations.subgraph(phoenix_start_stops_PointUid_unique)
pos=nx.get_node_attributes(G_phoenix_start_stops,'position')
nx.draw_networkx_nodes(G_phoenix_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
```




    <matplotlib.collections.PathCollection at 0x22c1b6933c8>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_19.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
phoenix_station_geometries_df=phoenix_lines_df4G[['PointUid','geometry_x','LineUid']] 
G_phoenix_bus_transfer,phoenix_transfer_df_concat=transfer_stations_network(phoenix_station_geometries_df,transfer_distance=200,transfer_weight_ratio=9,speed=45) 
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\pandas\core\frame.py:1554: FutureWarning: Using short name for 'orient' is deprecated. Only the options: ('dict', list, 'series', 'split', 'records', 'index') will be used in a future version. Use one of the above to silence this warning.
      FutureWarning,
    100%|██████████| 831/831 [00:05<00:00, 139.23it/s]
    


```python
G_phoenix_bus=nx.compose(G_phoenix_bus_stations,G_phoenix_bus_transfer)
phoenix_edge_color_=list(nx.get_edge_attributes(G_phoenix_bus, 'time_cost').values()) #'forward_length'
# edge_color=[i/max(edge_color_) for i in edge_color_]
ax=G_draw(G_phoenix_bus,edge_color=phoenix_edge_color_,labels=nx.get_node_attributes(G_phoenix_bus_stations,'station_name'),font_size=20,figsize=(20, 20))
nx.draw_networkx_nodes(G_phoenix_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
phoenix_pos_edge=nx.get_node_attributes(G_phoenix_bus_transfer,'position')
nx.draw_networkx_edges(G_phoenix_bus_transfer,pos=phoenix_pos_edge,edgelist=G_phoenix_bus_transfer.edges,edge_color='red',ax=ax)
```




    <matplotlib.collections.LineCollection at 0x22c04199108>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_20.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
phoenix_all_shortest_path_dict,phoenix_all_shortest_length_dict=bus_shortest_paths(G_phoenix_bus,phoenix_start_stops_PointUid)
```

    100%|██████████| 45/45 [00:00<00:00, 92.50it/s]
    


```python
phoenix_shortest_paths_list=list(phoenix_all_shortest_path_dict.values())[10]
phoenix_shortest_path=list(phoenix_shortest_paths_list.values())[300]
phoenix_H=G_phoenix_bus.subgraph(phoenix_shortest_path)
ax=G_draw(phoenix_H,edge_color=list(nx.get_edge_attributes(phoenix_H, 'time_cost').values()),labels=nx.get_node_attributes(phoenix_H,'station_name'),font_size=20,figsize=(10, 10),with_labels=True)
nx.draw_networkx_nodes(G_phoenix_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
```




    <matplotlib.collections.PathCollection at 0x22c182dc8c8>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_21.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
phoenix_shortest_routes_fre_df,phoenix_shortest_df_dict=bus_service_index(G_phoenix_bus,phoenix_all_shortest_path_dict,phoenix_all_shortest_length_dict,start=0,end=25,step=5)   
phoenix_shortest_routes_fre_df_statistics=phoenix_shortest_routes_fre_df.describe(include='all')
phoenix_shortest_routes_fre_df_statistics
```

    100%|██████████| 45/45 [00:00<00:00, 230.47it/s]
    




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
      <th>5</th>
      <th>10</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>549.755556</td>
      <td>273.711111</td>
      <td>7.533333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>34.052203</td>
      <td>31.372123</td>
      <td>3.596716</td>
    </tr>
    <tr>
      <th>min</th>
      <td>461.000000</td>
      <td>234.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>536.000000</td>
      <td>256.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>558.000000</td>
      <td>267.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>568.000000</td>
      <td>285.000000</td>
      <td>9.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>592.000000</td>
      <td>356.000000</td>
      <td>21.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
phoenix_shortest_routes_fre_df_,phoenix_shortest_df_dict_=bus_service_index(G_phoenix_bus,phoenix_all_shortest_path_dict,phoenix_all_shortest_length_dict,start=0,end=25,step=1)   
phoenix_shortest_routes_fre_df_statistics_=phoenix_shortest_routes_fre_df_.describe(include='all')
phoenix_shortest_routes_fre_df_statistics_
```

    100%|██████████| 45/45 [00:00<00:00, 152.55it/s]
    




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
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>2</th>
      <th>8</th>
      <th>7</th>
      <th>1</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>13</th>
      <th>14</th>
      <th>12</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>45.000000</td>
      <td>15.000000</td>
      <td>34.000000</td>
      <td>32.000000</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>154.044444</td>
      <td>182.133333</td>
      <td>130.777778</td>
      <td>89.533333</td>
      <td>61.200000</td>
      <td>58.933333</td>
      <td>75.866667</td>
      <td>21.600000</td>
      <td>33.577778</td>
      <td>15.800000</td>
      <td>4.333333</td>
      <td>1.733333</td>
      <td>1.764706</td>
      <td>1.625000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23.877486</td>
      <td>14.589224</td>
      <td>10.226723</td>
      <td>10.963991</td>
      <td>14.407384</td>
      <td>11.054411</td>
      <td>13.325642</td>
      <td>7.155418</td>
      <td>11.364796</td>
      <td>9.663897</td>
      <td>3.261344</td>
      <td>0.457738</td>
      <td>0.430562</td>
      <td>0.491869</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>102.000000</td>
      <td>159.000000</td>
      <td>102.000000</td>
      <td>71.000000</td>
      <td>41.000000</td>
      <td>39.000000</td>
      <td>62.000000</td>
      <td>6.000000</td>
      <td>15.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>140.000000</td>
      <td>167.000000</td>
      <td>128.000000</td>
      <td>81.000000</td>
      <td>49.000000</td>
      <td>52.000000</td>
      <td>66.000000</td>
      <td>16.000000</td>
      <td>23.000000</td>
      <td>10.000000</td>
      <td>3.000000</td>
      <td>1.500000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>151.000000</td>
      <td>183.000000</td>
      <td>133.000000</td>
      <td>90.000000</td>
      <td>55.000000</td>
      <td>60.000000</td>
      <td>70.000000</td>
      <td>26.000000</td>
      <td>39.000000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>168.000000</td>
      <td>193.000000</td>
      <td>137.000000</td>
      <td>98.000000</td>
      <td>74.000000</td>
      <td>70.000000</td>
      <td>83.000000</td>
      <td>27.000000</td>
      <td>43.000000</td>
      <td>20.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>194.000000</td>
      <td>207.000000</td>
      <td>147.000000</td>
      <td>115.000000</td>
      <td>87.000000</td>
      <td>75.000000</td>
      <td>106.000000</td>
      <td>31.000000</td>
      <td>49.000000</td>
      <td>41.000000</td>
      <td>17.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib.pyplot as plt
phoenix_shortest_routes_fre_df_statistics_.T.sort_index().plot(y='mean')
plt.xticks(phoenix_shortest_routes_fre_df_statistics_.T.index)
plt.show()
```


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_22.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
phoenix_paths_0=list(phoenix_shortest_df_dict.values())[0]
G_draw_range_paths(G_phoenix_bus,phoenix_paths_0,figsize=(10, 10),node_size=10)
```

    Graph has 592 nodes with 591 paths(edges)
    Graph has 394 nodes with 393 paths(edges)
    Graph has 38 nodes with 37 paths(edges)
    


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_23.png" height='auto' width='auto' title="caDesign"></a> 
    



    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_24.png" height='auto' width='auto' title="caDesign"></a> 
    



    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_25.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
phoenix_edges_dict,ax=G_draw_paths_composite(G_phoenix_bus,phoenix_paths_0,figsize=(15,15),node_size=10)
nx.draw_networkx_nodes(G_phoenix_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
```

    [ 5 10 15]
    Graph has 831 nodes with 830 paths(edges)
    




    <matplotlib.collections.PathCollection at 0x22c5d201e88>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_26.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
nx.write_gpickle(G_phoenix_bus, "./results/G_phoenix_bus.gpickle")
```


```python
phoenix_subway_stations=postSQL2gpd(table_name='phoenix_subway_stations',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')
phoenix_subway_routes=postSQL2gpd(table_name='phoenix_subway_lines',geom_col='geometry',myusername='postgres',mypassword='123456',mydatabase='workshop-LA-UP_IIT')

G_phoenix_subway_stations,phoenix_subway_s_e_nodes,phoenix_subway_start_stops_PointUid,phoenix_subway_lines_df4G=bus_network(phoenix_b_centroid,phoenix_subway_stations,phoenix_subway_routes,speed=60*2,start_stops_distance=1200) #bus_network_structure
G_phoenix_subway_stations.remove_edges_from(phoenix_subway_s_e_nodes)
G_draw(G_phoenix_subway_stations,edge_color=list(nx.get_edge_attributes(G_phoenix_subway_stations, 'time_cost').values()),labels=nx.get_node_attributes(G_phoenix_subway_stations,'station_name'),font_size=20,figsize=(10, 10),with_labels=True)
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\sqlalchemy\dialects\postgresql\base.py:3199: SAWarning: Did not recognize type 'geometry' of column 'geometry'
      "Did not recognize type '%s' of column '%s'" % (attype, name)
    100%|██████████| 2/2 [00:00<00:00, 90.88it/s]

    __________________________________________________
    The data has been read from PostSQL database...
    __________________________________________________
    The data has been read from PostSQL database...
    

    
    




    <AxesSubplot:>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_27.png" height='auto' width='auto' title="caDesign"></a> 
    



```python
G_phoenix_subway_bus_transfer,phoenix_subway_bus_transfer_df_concat=transfer_stations_network_subway_bus(bus_df=phoenix_lines_df4G,subway_df=phoenix_subway_lines_df4G,transfer_distance=300,transfer_weight_ratio=9,speed=45)   #2 
G_phoenix_subway_bus=nx.compose(G_phoenix_bus,G_phoenix_subway_stations)
G_phoenix_subway_bus_transfer=nx.compose(G_phoenix_subway_bus,G_phoenix_subway_bus_transfer)
```

    C:\Users\richi\anaconda3\envs\earthpy\lib\site-packages\pandas\core\frame.py:1554: FutureWarning: Using short name for 'orient' is deprecated. Only the options: ('dict', list, 'series', 'split', 'records', 'index') will be used in a future version. Use one of the above to silence this warning.
      FutureWarning,
    100%|██████████| 12/12 [00:00<00:00, 126.65it/s]
    


```python
phoenix_all_shortest_path_dict_,phoenix_all_shortest_length_dict_=bus_shortest_paths(G_phoenix_subway_bus_transfer,phoenix_start_stops_PointUid)
phoenix_shortest_routes_fre_df_,phoenix_shortest_df_dict_=bus_service_index(G_phoenix_subway_bus_transfer,phoenix_all_shortest_path_dict_,phoenix_all_shortest_length_dict_,start=0,end=25,step=5) 
```

    100%|██████████| 45/45 [00:00<00:00, 90.35it/s]
    100%|██████████| 45/45 [00:00<00:00, 231.35it/s]
    


```python
phoenix_paths_0_=list(phoenix_shortest_df_dict_.values())[0]
phoenix_edges_dict_,ax=G_draw_paths_composite(G_phoenix_subway_bus_transfer,phoenix_paths_0_,figsize=(15,15),node_size=10)
nx.draw_networkx_nodes(G_phoenix_start_stops,pos=pos,node_size=100,ax=ax,node_color='red')
```

    [ 5 10 15]
    Graph has 831 nodes with 830 paths(edges)
    




    <matplotlib.collections.PathCollection at 0x22c5ee27c08>




    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_28.png" height='auto' width='auto' title="caDesign"></a> 


#### 1.3.2 The result comparison
Under different time cost sections, the shortest path's mean difference is 2.5 times for 5 minutes, 10.0 times for 10 minutes, and 2.8 times for 15 minutes. Compared with Phoenix Hill, Qingliang Mountain has higher accessibility, and the difference is noticeable. 


```python
shortest_routes_fre_difference=shortest_routes_fre_df_statistics/phoenix_shortest_routes_fre_df_statistics
shortest_routes_fre_difference
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
      <th>5</th>
      <th>10</th>
      <th>15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.733333</td>
      <td>1.733333</td>
      <td>1.733333</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.485837</td>
      <td>10.012756</td>
      <td>2.760381</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.966845</td>
      <td>3.072072</td>
      <td>2.792693</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.555315</td>
      <td>11.247863</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.342351</td>
      <td>10.359375</td>
      <td>2.600000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.559140</td>
      <td>10.033708</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.570423</td>
      <td>10.000000</td>
      <td>2.555556</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.496622</td>
      <td>8.151685</td>
      <td>2.285714</td>
    </tr>
  </tbody>
</table>
</div>



In both cases, the number of the shortest path of time cost increases first and then decreases, but the turning point of the maximum number is different. Phoenix Hill is located in the 6-minute zone with more accessibility, while Qingliang Mountain is located in the 4-minute zone.


```python
import matplotlib.pyplot as plt
ax=phoenix_shortest_routes_fre_df_statistics_.T.sort_index().plot(y='mean')
shortest_routes_fre_df_statistics_.T.sort_index().plot(y='mean',ax=ax)
plt.xticks(shortest_routes_fre_df_statistics_.T.index)
plt.show()
```


    
<a href=""><img src="./workshop-LA-UP_IIT/imgs/publicNetwork_29.png" height='auto' width='auto' title="caDesign"></a> 
    

