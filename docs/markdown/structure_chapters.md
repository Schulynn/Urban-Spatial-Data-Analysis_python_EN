# Chapter structure thinking -- uncertainty
In line with people's learning habits, the book's chapter structure will make the learning process smooth. Still, it must be not one step in place in the writing process, but a constant adjustment and integration process, so I added this part to think about this problem.

### Knowledge point distribution



```mermaid
classDiagram

单个分类POI数据爬取与地理空间点地图 --> 数据分析 : 
单个分类POI数据爬取与地理空间点地图 : 单个分类POI爬取
单个分类POI数据爬取与地理空间点地图 : 将csv格式的POI数据转换为pandas的DataFrame
单个分类POI数据爬取与地理空间点地图 : 将数据格式为DataFramed的POI数据转换为GeoPandas地理空间数据GeoDataFrame
单个分类POI数据爬取与地理空间点地图 : 使用plotly库建立地图

多个分类POI数据爬取与描述性统计 --|> 数据分析
多个分类POI数据爬取与描述性统计 --|> 描述性统计 : 拆分
多个分类POI数据爬取与描述性统计 : 多个分类POI爬取
多个分类POI数据爬取与描述性统计 : 批量转换.csv格式数据为GeoDataFrame

描述性统计 --|> 统计学知识点
描述性统计 : 数据种类
描述性统计 : 集中量数与变异量数
描述性统计 : 频数（次数）分布表和直方图
描述性统计 : 中位数
描述性统计 : 算数平均数
描述性统计 : 标准差
描述性统计 : 标准计分

正态分布与概率密度函数_异常值处理 --|> 统计学知识点
正态分布与概率密度函数_异常值处理 : 正态分布
正态分布与概率密度函数_异常值处理 : 概率密度函数
正态分布与概率密度函数_异常值处理 : 累积分布函数
正态分布与概率密度函数_异常值处理 : 偏度与峰度
正态分布与概率密度函数_异常值处理 : 检验数据集是否服从正态分布
正态分布与概率密度函数_异常值处理 : 异常值处理
正态分布与概率密度函数_异常值处理 : 给定特定值计算概率，以及找到给定概率的值

OSM数据处理 --|> 数据分析
OSM数据处理 : OSM原始数据处理
OSM数据处理 : 读取、转换.osm数据

核密度估计与地理空间点密度分布 --|> 数据分析
核密度估计与地理空间点密度分布 --|> 统计学知识点
核密度估计与地理空间点密度分布 : *核密度估计
核密度估计与地理空间点密度分布 : 核密度估计结果转换为地理栅格数据

标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 --|> 统计学知识点
标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 --|> 公共健康数据的地理空间分布 : 拆分
公共健康数据的地理空间分布 --|> 数据分析
标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 :  标准误
标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 :  中心极限定理
标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 :  t分布
标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 :  统计显著性
标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 :  效应量
标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 :  置信区间
标准误_中心极限定理_t分布_统计显著性_效应量_置信区间_公共健康数据的地理空间分布与相关性分析 :  相关性分析

回归公共健康数据与梯度下降法 --|> 统计学知识点
回归公共健康数据与梯度下降法  : 公共健康数据的简单线性回归
回归公共健康数据与梯度下降法  : k-NN（k-近邻模型）
回归公共健康数据与梯度下降法  : MAE平均绝对误差与MSE均方误差
回归公共健康数据与梯度下降法  : 公共健康数据的多元回归
回归公共健康数据与梯度下降法  : 多项式回归
回归公共健康数据与梯度下降法  : 正则化
回归公共健康数据与梯度下降法  : 梯度下降法
回归公共健康数据与梯度下降法 --|> 回归公共健康数据 : 拆分
回归公共健康数据 --|> 数据分析

```


### Knowledge associated

```mermaid
erDiagram
    USDAM-py ||--o{ geospatial-data : place 
    USDAM-py }|..|{ non : non
    data ||--|{ POI : Baidu
    POI ||--|{ geospatial-data : get-data
    POI ||--|{ single-classificaiton : crawler
    POI ||--|{ multiple-classificaiton : crawler
    single-classificaiton ||--|{ geospatial-data : geo
    USDAM-py }|..|{ descriptive-statistics : knowledge-point
    multiple-classificaiton ||--|| descriptive-statistics : knowledge-point
```