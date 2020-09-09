> Created on Created on Sat Jul  4 10/29/03 2020 @author: Richie Bao-caDesign设计(cadesign.cn) 

## 1. Normal distribution and probability density function, outlier handling
In the previous section, the calculation function of `frequency_bins(df,bins)` frequency distribution is defined, and the diagram is printed in the form of a histogram to view the data change. However, when the interval is determined, the interval is adjusted according to experience to observe the distribution of data and calculate the percentage of all interval price frequencies. Is there a way to determine that this set of data is normally distributed? If so,  is it possible to accurately calculate the probability of a normal distribution for a given price, or the proportion of a given price interval?

First, the frequency is calculated when the group spacing is 5. Place the frequency distribution calculation function in a common `util_poi` file for easy invocation.


```python
import util_poi
import pandas as pd
poi_gpd=pd.read_pickle('./data/poiAll_gpd.pkl') #Read the POI data already saved in .pkl format, including the 'geometry' field, as the GeoDataFrame geographic information data, which can be quickly viewed through `poi_gpd.plot().` 
delicacy_price=poi_gpd.xs('poi_0_delicacy',level=0).detail_info_price  #Extract food price data
delicacy_price_df=delicacy_price.to_frame(name='price').astype(float) 
bins=list(range(0,300+5,5))+[600] #The segmentation interval (group distance /spacing) is configured because the histogram is more comfortable to interpret the values with small differences. The price distribution standard deviation of food is more extensive, namely, the degree of dispersion. Therefore, when dividing the group distance, the part far from the center value is not divided by equidistance.
poiPrice_fre=util_poi.frequency_bins(delicacy_price_df,bins) 
print(poiPrice_fre.head())
```

          index  fre    relFre  median  fre_percent%
    0    [0, 5)    0  0.000000     NaN      0.000000
    1   [5, 10)    8  0.010191   101.5      1.019108
    2  [10, 15)   42  0.053503   107.0      5.350318
    3  [15, 20)   85  0.108280   111.0     10.828025
    4  [20, 25)   75  0.095541   116.0      9.554140
    

### 1.1 Normal distribution and probability density function
#### 1.1.1 Normal distribution 
Before we go on to the probability density function,  let us think about the normal distribution, is also known as the Gaussian distribution. The common name for a normal distribution is the bell curve because of its shape. Use the `numpy.random.normal()` method in the NumPy library to generate one-dimensional data that meets the specified mean, standard deviation (you can also use 'norm' method provided in the Scipy library stats class), and print the histogram, and the corresponding distribution curve calculated by the probability density function. The y-axis of the figure below represents the frequency of randomly generated values (because the density=Ture is configured, the returned frequency is the result after normalization). The x-axis is the numerical distribution of the randomly generated dataset. Because the `bins=30` in the `plt.hist()` is set, the value is divided into 30 bins, and the frequency of each bin is calculated. The vertex position of the curve is mean, median, and mode. And the frequency is the highest, the corresponding value is 0, which is also mode. You can redefine the value of the mean 'mu' parameter, and the curve vertex will change accordingly. When the mode moves to both sides, the curve's height drops, indicating that the occurrence of these values gradually decreases, that is, the frequency decreases. Statistically, the figure below can be described as follows: x follows a normal distribution with an average value of 0 and a standard deviation of 0.1.

According to the shape of a normal distribution, its three basic properties are as follows: first, it is symmetric, meaning that the left and right parts are mirror images of each other, the mean, median, and mode are in the same position, and at the center of the distribution, the apex of the bell shape. The curve's center is the highest, and both ends of the curve tilt downward, showing a single peak. Third, the normal distribution is asymptotic, which means that the left tail and the right tail of the distribution never touch the bottom line, the x-axis.

The normal distribution is of great significance; nature and human society often appear normal distribution of all kinds of data, such as income level in the economy, people's intelligence quotient (IQ) scores and test scores, affected by a large number of small random disturbance in the natural world, so can according to this phenomenon infer that accurate probability of some situation. Simultaneously, it should be noted that the normal distribution is the so-called theoretical distribution in statistics, that is, few values strictly follow the normal distribution, but approximate to, and maybe far from. If the normal distribution assumption is violated, the calculated probability results based on the normal distribution assumption will no longer be valid.

> Knowledge of statistics are usually based on the published book or teaching material as the basis for interpretation, and to according to the contents of this paper, with the python language as a tool to analyze the change of the data, which can be directly application code to solve the related problems, and be able to more in-depth understanding through the analysis of the specific data statistics related knowledge. In this section, reference: Timothy C.Urdan.Statistics in Plain English.   Routledge; 3rd edition (May 27, 2010).


```python
#The following cases refer to the `numpy.random.normal` case on SciPy.org.
import numpy as np
import math
#Draw samples from the distribution
mu, sigma = 0, 0.1 #mean and standard deviation
s=np.random.normal(mu, sigma, 1000)
#Verify the mean and the variance
print("mean<0.01:",abs(mu - np.mean(s)) < 0.01)
print("sigmag variance<0.01",abs(sigma - np.std(s, ddof=1)) < 0.01)
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, bins=30,density=True) #Paramter，density=True,   If True, the first element of the return tuple will be the counts normalized to form a probability density
PDF=1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (bins - mu)**2 / (2 * sigma**2) ) #The value calculation formula of y-direction is the probability density function.
plt.plot(bins,PDF ,linewidth=2, color='r',label='PDF') 

CDF=PDF.cumsum() #Calculate the cumulative distribution function
CDF/=CDF[-1] #The CDF value distribution is adjusted to the interval of [0,1] by dividing the maximum value.
plt.plot(bins,CDF,linewidth=2, color='g',label='CDF') 

plt.legend()
plt.show()
```

    mean<0.01: True
    sigmag variance<0.01 True
    


<a href="https://jupyter.org/"><img src="./imgs/3_1.png" height="auto" width="auto" title="caDesign"/></a>


#### 1.1.2 Probability density function，PDF
When the histogram's spacing is reduced to the limit, a curve can be fitted, and the formula to calculate the distribution curve is the probability density function:$f(x)= \frac{1}{ \sigma  \times  \sqrt{2 \pi } }  \times  e^{- \frac{1}{2}  [ \frac{x- \mu }{  \sigma  } ]^{2} } $ Where, $\sigma$  is the standard deviation; $\mu$ is the average; $e$ is the base of the natural logarithm, whose value is about 2.7182... In the above code, the calculation of probability density function is not each numerical calculation; it uses `Plt.hist()` return values 'bins' instead of, in other words, the left edge and right edge of each group are divided into 30 parts, so the first digit is the left edge of the first bin, the last digit is the right edge of the last bin, and all the middle edges overlap the left and the right edges. And the integral of the probability density function is the cumulative distribution function(CDF). You can compute with `numpy.cumsum()` for the cumulative sum of array elements on a given axis. 

Chart print library mainly include [Matplotlib](https://matplotlib.org/)，[plotly(including dash)](https://plotly.com/)，[bokeh](https://docs.bokeh.org/en/latest/index.html)，[seaborn](https://seaborn.pydata.org/).There is no set standard for which chart to print. Still, it is usually determined based on the purpose of the current data chart to print, which library meets the requirements, and which library an individual is more comfortable with to use.  When the density function curve is printed using the Matplolib library, it is calculated by itself. Seaborn provides the method `seaborn.distplot(),`  and the described results can be obtained directly after specifying the 'bins' parameters. It is necessary to pay attention to the configuration of the 'bins' parameter. If it is an integer value, it is the number dividing the bin(box) of the same width; If it is a list, it is the 'bin' edge, for example, [1,2,3,4], says [[1,2),[2,3),[3,4]] bandwidth(BW) list, `[` represents the data that contains the left edge, `]` represents the data that contains the right edge, and `(`, and `)` means the data that does not hold the left or right edge, respectively. With pandas `pandas.core.indexes.range.RangeIndex` data format. 

Return to the experimental data, directly calculate and print the probability density function of the POI food price (vertical axis), and connect to the curve(distribution curve). To further observe, the fitting degree with the curve when the group spacing is continuously reduced. The histogram under the change of successive group spacing is printed circulatively to observe the histogram's change.


```python
import seaborn as sns
import matplotlib.pyplot as plt
import math
sns.set(style="white", palette="muted", color_codes=True)
bins=list(range(0,50,5))[1:]
bin_num=len(bins)

ncol=5
nrow=math.ceil(bin_num/ncol)
fig, axs = plt.subplots(ncols=ncol,nrows=nrow,figsize=(30,10),sharex=True)
ax=[(row,col) for row in range(nrow) for col in range(ncol)]
i=0
for bin in bins:
    sns.distplot(delicacy_price_df.price, bins=bin, color="r",ax=axs[ax[i]],kde = True,norm_hist=True).set(title="bins=%d"%bin)
    i+=1
```


<a href="https://jupyter.org/"><img src="./imgs/3_2.png" height="auto" width="auto" title="caDesign"/></a>


#### 1.1.3 Skew and kurtosis
Two concepts describe the characteristics of normal distribution(or the distribution curve of probability density function). One is skew, and the other is kurtosis. 

Data with a positive skew and negative skew attributes are established by the Scipy library 'skewnorm' method, and the 'skew' value is calculated by the 'skew' method. The formula is as  follows: $skewness= \frac{3( \ \mu  -median)}{ \sigma } $ Where, $\mu$ is the mean value; $sigma$ is the standard deviation. If the skew value is negative, namely negative skew, the probability density function's left tail is longer than the right tail. The skew value is positive, that is, positive skew, and the tail on the right side of the probability density function is longer than the left side.

Numpy was used to establish data with an attribute of peak distribution(thin tail) and flat peak distribution(thick tail), configure its absolute parameter to be 0, change the standard deviation, and use the Scipy 'kurtosis' method to calculate its kurtosis value(there are several formulas for kurtosis calculation, and there are also differences in the formulas of different software platforms). When the mean value is 0, and the standard deviation is 1, the normal distribution is the standard normal distribution, and the formula of probability density function can be simplified as:$f(x)= \frac{1}{   \sqrt{2 \pi } }  \times  e^{- \frac{ x^{2}}{2}}$

> Note that when Numpy is used to generate the dataset, the parameter of variance $ \sigma ^{2} $ is directly used, and the formula of probability density function can be adjusted as follows:$f(x |  \mu ,  \sigma ^{2} )= \frac{1}{ \sqrt{2 \pi } \sigma ^{2} }  \times  e^{- \frac{ (x- \mu )^{2} }{2\sigma ^{2} }  } $, $\sigma ^{2}$ the variance：  $\sigma ^{2} =   \frac{1}{N}   \sum_{i=1}^N { ( x_{i}- \mu ) ^{2} } $


```python
#Establishes data with positive skew, and negative skew attributes
from scipy.stats import skewnorm
from scipy.stats import skew
import matplotlib.pyplot as plt
skew_list=[7,-7]
skewNorm_list=[skewnorm.rvs(a, scale=1,size=1000,random_state=None) for a in skew_list]
skewness=[skew(d) for d in skewNorm_list]
print('skewness for data:',skewness) #Verify the skew value

#Establish the data with the attributes of peak distribution, and flat peak distribution
import numpy as np
import math
from scipy.stats import kurtosis
mu_list=[0,0,0,]
variance=[0.2,1.0,5.0,]
normalDistr_paras=zip(mu_list,[math.sqrt(sig2) for sig2 in variance])#Configure multiple mean and standard deviation pairs
s_list=[np.random.normal(para[0], para[1], 1000) for para in normalDistr_paras]
kur=[kurtosis(s,fisher=True) for s in s_list]
print("kurtosis for data:",kur)

i=0
for skew_data in skewNorm_list:
    sns.kdeplot(skew_data,label="skewness=%.2f"%skewness[i])
    i+=1

n=0
for s in s_list:
    sns.kdeplot(s,label="μ=%s, $σ^2$=%s=%s,kur=%.2f"%(mu_list[n],variance[n],kur[n]))
    n+=1  
plt.plot()
```

    skewness for data: [0.90957801236704, -0.9760661360942957]
    kurtosis for data: [0.21802304789205973, 0.03826633358066145, 0.036763247950971856]
    



<a href="https://jupyter.org/"><img src="./imgs/3_3.png" height="auto" width="auto" title="caDesign"/></a>


Return the experimental data and calculate the skew and kurtosis of POI food data price. The computed value of skew is positive and is positive skew. The tail on the right side of the probability density function is longer than the left, indicating that most prices are lower, and a few higher prices hold the distributed tail to the other end.  Kurtosis is the peak distribution.


```python
delicacy_price_df_clean=delicacy_price_df.dropna()
print("skewness:%.2f, kurtosis:%.2f"%(skew(delicacy_price_df_clean.price),kurtosis(delicacy_price_df_clean.price,fisher=True)))
```

    skewness:3.96, kurtosis:32.31
    




#### 1.1.4 Verify that the dataset is normally distributed
First, the standard score is calculated, and the mean value and standard deviation of the standardized price dataset are 0 and 1. The distribution curve of the probability density function  (theoretical value) is drawn. Simultaneously, to superpose the curve, which meets the above conditions, is randomly generated by Numpy and meets the distribution curve(observed value) of the normally distributed dataset. You can compare the differences. Because the experimental data, namely the price of delicious food, is positively skewed, it can be observed that there is a difference in the data on the left side of the mean; in contrast, the trend on the right side is consistent. At the same time, it has a higher kurtosis than the standard normal distribution curve. Because this part of code is then called to compare the dataset to the standard normal distribution, it is defined as a function.


```python
def comparisonOFdistribution(df,field,bins=100):
    import pandas as pd
    import numpy as np
    import seaborn as sns
    '''
    funciton-The distribution curve of the probability density function ( observed/empirical data) of the z-score  dataset was compared with the standard normal distribution(theoretical set) 
    
    Params:
    df - Contains DataFrame format type data for the dataset to be analyzed
    field - Specifies the column name of the df data being analyzed
    bins - Specifies the bandwidth with a single integer representing the number of the bin, or a list representing a list of multiple bandwidths
    '''
    df_field_mean=df[field].mean()
    df_field_std=df[field].std()
    print("mean:%.2f, SD:%.2f"%(df_field_mean,df_field_std))

    df['field_norm']=df[field].apply(lambda row: (row-df_field_mean)/df_field_std) #Standardized price(standard score, z-score), or use the method `from scipy.stats import zscore` used in the previous section

    #Verify the z-score, the standardized mean must be 0, and the standard deviation must be 1.0
    df_fieldNorm_mean=df['field_norm'].mean()
    df_fieldNorm_std=df['field_norm'].std()
    print("norm_mean:%.2f, norm_SD:%.2f"%(df_fieldNorm_mean,df_fieldNorm_std))
  
    sns.distplot(df['field_norm'], bins=bins,)

    s=np.random.normal(0, 1, len(df[field]))
    sns.distplot(s, bins=bins)
comparisonOFdistribution(delicacy_price_df_clean,'price',bins=100)    
```

    mean:53.58, SD:44.12
    norm_mean:-0.00, norm_SD:1.00
    

    C:\Users\richi\conda\envs\pyG\lib\site-packages\ipykernel_launcher.py:17: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
    


<a href="https://jupyter.org/"><img src="./imgs/3_4.png" height="auto" width="auto" title="caDesign"/></a>


* Outlier handling

The skewness difference can be observed from the histogram or box diagram that a few parts of the higher price appear. Assuming that this part's price is an outlier, what will happen to the distribution curve of the dataset after the outlier is removed?

When dealing with outliers, first create a simple dataset to understand outliers and find the appropriate method. First reference:Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and Handle Outliers", The ASQC Basic References in Quality Control: Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. Using the method provided herein, the formula is:

$MAD= median_{i}{ \{ |  x_{i}-   \widetilde{x} |\} }$ ，among them,MAD（the median of the absolute deviation about the median /Median Absolute Deviation）is the absolute median difference，where,$\widetilde{x}$ is the median；

$M_{i}= \frac{0.6745( x_{i} - \widetilde{x} )}{MAD}$  ，where $M_{i}$is the modified z-score(modified  z-score)，$\widetilde{x}$ is the median。

According to the calculation results below, this formula can effectively judge outliers.


```python
import numpy as np
import matplotlib.pyplot as plt
outlier_data=np.array([2.1,2.6,2.4,2.5,2.3,2.1,2.3,2.6,8.2,8.3]) #Create a simple dataset for easy observation
ax1=plt.subplot(221)
ax1.margins(0.05) 
ax1.boxplot(outlier_data)
ax1.set_title('simple data before')

def is_outlier(data,threshold=3.5):
    import numpy as np
    '''
    function-Judge outliers
        
    Params:
    data - Data to be analyzed, either as a list or as a one-dimensional array
    threshold - A boundary condition that determines whether an outlier is an outlier    
    '''
    MAD=np.median(abs(data-np.median(data)))
    modified_ZScore=0.6745*(data-np.median(data))/MAD
    #print(modified_ZScore)
    is_outlier_bool=modified_ZScore>threshold    
    return is_outlier_bool,data[~is_outlier_bool]
    
is_outlier_bool,data_clean=is_outlier(outlier_data,threshold=3.5)    
print(is_outlier_bool,data_clean)

ax2=plt.subplot(222)
ax2.margins(0.05) 
ax2.boxplot(data_clean)
ax2.set_title('simple data after')

_,delicacyPrice_outliersDrop=is_outlier(delicacy_price_df_clean.price,threshold=3.5)
print("Description of original data:",delicacy_price_df_clean.price.describe())
print("-"*50)
print("Description of data after outlier processing:",delicacyPrice_outliersDrop.describe())

ax3=plt.subplot(223)
ax3.margins(0.05) 
ax3.boxplot(delicacy_price_df_clean.price)
ax3.set_title('experimental data before')

ax3=plt.subplot(224)
ax3.margins(0.05) 
ax3.boxplot(delicacyPrice_outliersDrop)
ax3.set_title('experimental data after')

plt.show()
```

    [False False False False False False False False  True  True] [2.1 2.6 2.4 2.5 2.3 2.1 2.3 2.6]
    Description of original data: count    785.000000
    mean      53.584076
    std       44.123529
    min        5.000000
    25%       23.000000
    50%       43.000000
    75%       72.000000
    max      571.000000
    Name: price, dtype: float64
    --------------------------------------------------
    Description of data after outlier processing: count    770.000000
    mean      49.766883
    std       31.433613
    min        5.000000
    25%       23.000000
    50%       42.000000
    75%       71.000000
    max      159.000000
    Name: price, dtype: float64
    


<a href="https://jupyter.org/"><img src="./imgs/3_5.png" height="auto" width="auto" title="caDesign"/></a>



```python
comparisonOFdistribution(pd.DataFrame(delicacyPrice_outliersDrop,columns=['price']),'price',bins=100)
```

    mean:49.77, SD:31.43
    norm_mean:0.00, norm_SD:1.00
    


<a href="https://jupyter.org/"><img src="./imgs/3_6.png" height="auto" width="auto" title="caDesign"/></a>


If the higher price is removed as an outlier, the defined `comparisonOFdistribution(df, field,bins=100)` function is called to print the extracted value's probability density function. By comparing the standard normal distribution, it can be found that the higher value part has changed on the right side, while the other parts have not changed significantly. So how do you verify that your dataset is normally distributed? The Scipy library provides various normality testing methods: `kstest`, `shapiro`, `normaltest`, `anderson`.  The  p-value displayed by the calculation result is 0, namely p-value<0.05, reject the null hypothesis; after cleaning the outlier, the food price dataset does not obey the normal distribution; that is, the non-normal dataset. Only a normal distribution can calculate the probability of selecting a specific value or range from a population. This food price dataset is positive bias and peak distribution and is a non-normal dataset. The probability of normal distribution cannot be well applied to this kind of dataset.


```python
from scipy import stats
kstest_test=stats.kstest(delicacyPrice_outliersDrop,cdf='norm')
print(kstest_test)

shapiro_test=stats.shapiro(delicacyPrice_outliersDrop) #Official documentation notes, when N>5000, only the statistics are correct, but the p-value is not necessarily accurate. In this experiment, the data volume of `len(delicacyPrice_outliersDrop)` was 770 and could be used.
print("shapiroResults(statistic=%f,pvalue=%f)"%(shapiro_test))

normaltest_test=stats.normaltest(delicacyPrice_outliersDrop,axis=None)
print(normaltest_test)

anderson_test=stats.anderson(delicacyPrice_outliersDrop,dist='norm')
print(anderson_test)
```

    KstestResult(statistic=0.9999997133484281, pvalue=0.0)
    shapiroResults(statistic=0.922036,pvalue=0.000000)
    NormaltestResult(statistic=81.9249012085564, pvalue=1.6226831906209503e-18)
    AndersonResult(statistic=16.928516711550174, critical_values=array([0.573, 0.653, 0.783, 0.913, 1.086]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ]))
    

#### 1.1.5 Calculate the probability for a given value, and find the value for a given probability
If the dataset is normally distributed, the calculation can be performed directly using 'norm' method provided in the Scipy library 'stats' class.  The following cases refer to Scipy official website. `norm.cdf()`,`norm.sf()`, and`norm.ppf()` are used to calculate values less than or equal to a specific value, greater than or equal to a specific value, or find value with a given probability.


```python
from scipy.stats import norm
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = norm.stats(moments='mvsk')  
print('mean, var, skew, kurt=',(mean, var, skew, kurt)) #Verify correlation statistics that conform to the standard normal distribution.
x=np.linspace(norm.ppf(0.01),norm.ppf(0.99), 100) #norm.ppf  Percent point function (inverse of cdf — percentiles)
ax.plot(x, norm.pdf(x), 'r-', lw=5, alpha=0.6, label='norm pdf')  #norm.pdf is the probability density function
rv=norm() #Fixed shape(skew and kurtosis), location(mean), and scale(standard deviation) parameters, namely, fixed values, are specified
ax.plot(x, rv.pdf(x), 'k-', lw=3, label='frozen pdf') #frozen distribution

vals = norm.ppf([0.001, 0.5, 0.999])  #Returns values with probabilities of 0.1%, 50% and 99.9%, default loc=0,scale=1
print("Verify whether the CDF return value is equal to or approximate to the PPF return value:",np.allclose([0.001, 0.5, 0.999], norm.cdf(vals)))

r=norm.rvs(size=1000) #Specifies the dataset size
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()
```

    mean, var, skew, kurt= (array(0.), array(1.), array(0.), array(0.))
    Verify whether the CDF return value is equal to or approximate to the PPF return value: True
    


<a href="https://jupyter.org/"><img src="./imgs/3_7.png" height="auto" width="auto" title="caDesign"/></a>



```python
#Define a fixed distribution of datasets if you need to compute probabilities.
import seaborn as sns
print("Using .cdf to calculate the probability that the value is less than or equal to 113 is: ",norm.cdf(113,100,12)) #pdf(x, loc=0, scale=1) Configure Loc(mean) and scale(standard deviation)
print("Using .sf to calculate the value greater than or equal to 113, the probability is: ",norm.sf(113,100,12)) 
print("It can be observed .cdf（<=113）probability result +.sf(>=113) probability result is:",norm.cdf(113,100,12)+norm.sf(113,100,12))
print("Using .ppf to find the value with a given probability value of 6.066975% is:",norm.ppf(0.86066975,100,12))
```

    Using .cdf to calculate the probability that the value is less than or equal to 113 is:  0.8606697525503779
    Using .sf to calculate the value greater than or equal to 113, the probability is:  0.13933024744962208
    It can be observed .cdf（<=113）probability result +.sf(>=113) probability result is: 1.0
    Using .ppf to find the value with a given probability value of 6.066975% is: 112.99999986204986


In traditional probability calculation, given z-score(standard score), the corresponding probability value can be obtained by looking up tables, which is rarely used anymore. To visualize the probability values. The area enclosed by the curve, the horizontal axis, and the vertical line through the corresponding probability density function value is the probability value, which can be observed more clearly by plotting the chart.

```python
def probability_graph(x_i,x_min,x_max,x_s=-9999,left=True,step=0.001,subplot_num=221,loc=0,scale=1):
    import math
    '''
    function - Normal distribution probability calculation and graphical representation
    
    Paras:
    x_i - The value of the probability to be predicted
    x_min - Dataset interval minimum
    x_max - Dataset interval maximum
    x_s - The second value with the predicted probability is greater than the x_i value.
    left - Whether to calculate the probability of less than or equal to, or greater than or equal to the specified value
    step - The steps of the dataset interval
    subplot_num - Print the ordinal number of a subgraph. For example, in 221, the first 2 represents the column, the second 2 represents the row, and the third is the ordinal of the subgraph, that is, there are 2 rows, 2 columns and 4 subgraphs in total, and 1 is the first subgraph.
    loc - The average
    scale - The standard deviation
    '''
    x=np.arange(x_min,x_max,step)
    ax=plt.subplot(subplot_num)
    ax.margins(0.2) 
    ax.plot(x,norm.pdf(x,loc=loc,scale=scale))
    ax.set_title('N(%s,$%s^2$),x=%s'%(loc,scale,x_i))
    ax.set_xlabel('x')
    ax.set_ylabel('pdf(x)')
    ax.grid(True)
    
    if x_s==-9999:
        if left:
            px=np.arange(x_min,x_i,step)
            ax.text(loc-loc/10,0.01,round(norm.cdf(x_i,loc=loc,scale=scale),3), fontsize=20)
        else:
            px=np.arange(x_i,x_max,step)
            ax.text(loc+loc/10,0.01,round(1-norm.cdf(x_i,loc=loc,scale=scale),3), fontsize=20)
        
    else:
        px=np.arange(x_s,x_i,step)
        ax.text(loc-loc/10,0.01,round(norm.cdf(x_i,loc=loc,scale=scale)-norm.cdf(x_s,loc=loc,scale=scale),2), fontsize=20)
    ax.set_ylim(0,norm.pdf(loc,loc=loc,scale=scale)+0.005)
    ax.fill_between(px,norm.pdf(px,loc=loc,scale=scale),alpha=0.5, color='g')
plt.figure(figsize=(10,10))
probability_graph(x_i=113,x_min=50,x_max=150,step=1,subplot_num=221,loc=100,scale=12)
probability_graph(x_i=113,x_min=50,x_max=150,step=1,left=False,subplot_num=223,loc=100,scale=12)
probability_graph(x_i=113,x_min=50,x_max=150,x_s=90,step=1,subplot_num=222,loc=100,scale=12)
probability_graph(x_i=90,x_min=50,x_max=150,step=1,subplot_num=224,loc=100,scale=20)
plt.show()
```

<a href="https://jupyter.org/"><img src="./imgs/3_8.png" height="auto" width="auto" title="caDesign"/></a>


### 1.2 key point
#### 1.2.1 data processing technique
* Statistical analysis tools such as normal distribution and probability density function - scipy and numpy libraries

-Generate datasets that are normally distributed，`s=np.random.normal(0, 1, len(df[field]))`

-Specifies the skew value to generate a skewed distribution random variable(A skew-normal random variable),`skewNorm_list=[skewnorm.rvs(a, scale=1,size=1000,random_state=None) for a in skew_list]`

-Calculate kurtosis value,`kur=[kurtosis(s,fisher=True) for s in s_list]`

-Calculate the skew，`print("skewness:%.2f, kurtosis:%.2f"%(skew(delicacy_price_df_clean.price),kurtosis(delicacy_price_df_clean.price,fisher=True)))` Kurtosis calculation is also included.

-Verify that it is normally distributed,

`kstest_test=stats.kstest(delicacyPrice_outliersDrop,cdf='norm')`
`shapiro_test=stats.shapiro(delicacyPrice_outliersDrop)`
`normaltest_test=stats.normaltest(delicacyPrice_outliersDrop,axis=None)`
`anderson_test=stats.anderson(delicacyPrice_outliersDrop,dist='norm')`

-Calculates the probability of a normally distributed dataset, or specifies a probability return value

`print("用.cdf计算值小于或等于113的概率为：",norm.cdf(113,100,12)) #pdf(x, loc=0, scale=1) 配置Loc(均值)和scale(标准差)`
`print("用.sf计算值大于或等于113待概率为：",norm.sf(113,100,12)) `
`print("可以观察到.cdf（<=113）概率结果+.sf(>=113)概率结果为：",norm.cdf(113,100,12)+norm.sf(113,100,12))`
`print("用.ppf找到给定概率值为98%的数值为：",norm.ppf(.98,100,12))`

* Graph of the probability density function

-Use matlotlib library，

`plt.plot(bins,PDF ,linewidth=2, color='r',label='PDF')`  
`plt.plot(bins,CDF,linewidth=2, color='g',label='CDF')` Cumulative distribution function

-Use seaborn library，

`sns.distplot(delicacy_price_df.price, bins=bin, color="r",ax=axs[ax[i]],kde = True,norm_hist=True).set(title="bins=%d"%bin)`
`sns.kdeplot(skew_data,label="skewness=%.2f"%skewness[i])`

* Other tools

-The cumulative sum of values along a given axis, and （numpy），CDF=PDF.cumsum()

-Up the values，nrow=math.ceil(bin_num/ncol)

#### 1.5.2 The newly created function tool

* funciton-dataset(observed/empirical data) compared with the standard normal distribution(theoretical set)，comparisonOFdistribution(df,field,bins=100)
* function-Judge outliers，is_outlier(data,threshold=3.5)

#### 1.5.3 The python libraries that are being imported


```python
import util_poi #Custom toolsets
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skewnorm
from scipy.stats import skew
from scipy.stats import kurtosis
```
