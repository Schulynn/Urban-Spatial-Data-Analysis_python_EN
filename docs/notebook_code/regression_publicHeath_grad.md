> Created on Sat Jul 25 11/41/43 2020  @author: Richie Bao-caDesign设计(cadesign.cn)

## 1.Regression of public health data, with gradient descent method
Public health data can be divided into three categories, geographic information data, disease data, and economic conditions data. In public health data, economic condition data is regarded as an independent variable, while disease data is regarded as a dependent variable. Usually, the independent variable is the cause. The dependent variable is the effect. The independent variable is the factor that can be changed, while the dependent variable is the factor that cannot be changed.
Sino-English Table of Public Health Data(Field Mapping Table):
```python
PubicHealth_Statistic_columns={'geographic information':
                                                      {'Community Area':'社区', 
                                                       'Community Area Name':'社区名',},
                               'disease':
                                        {'natality':{'Birth Rate':'出生率',
                                                       'General Fertility Rate':'一般生育率',
                                                       'Low Birth Weight':'低出生体重',
                                                       'Prenatal Care Beginning in First Trimester':'产前3个月护理', 
                                                       'Preterm Births':'早产',
                                                       'Teen Birth Rate':'青少年生育率',},
                                        'mortality':{'Assault (Homicide)':'攻击（杀人）',
                                                     'Breast cancer in females':'女性乳腺癌',
                                                     'Cancer (All Sites)':'癌症', 
                                                     'Colorectal Cancer':'结肠直肠癌',
                                                     'Diabetes-related':'糖尿病相关',
                                                     'Firearm-related':'枪支相关',
                                                     'Infant Mortality Rate':'婴儿死亡率', 
                                                     'Lung Cancer':'肺癌',
                                                     'Prostate Cancer in Males':'男性前列腺癌',
                                                     'Stroke (Cerebrovascular Disease)':'中风(脑血管疾病)',},
                                        'lead':{'Childhood Blood Lead Level Screening':'儿童血铅水平检查',
                                                'Childhood Lead Poisoning':'儿童铅中毒',},
                                                'infectious':{'Gonorrhea in Females':'女性淋病', 
                                                'Gonorrhea in Males':'男性淋病', 
                                                'Tuberculosis':'肺结核',},
                                'economic condition':
                                                   {'Below Poverty Level':'贫困水平以下', 
                                                    'Crowded Housing':'拥挤的住房', 
                                                    'Dependency':'依赖',
                                                    'No High School Diploma':'没有高中文凭', 
                                                    'Per Capita Income':'人均收入',
                                                    'Unemployment':'失业',},
                                }
```

> References for this part
> 1. Cavin Hackeling. Mastering Machine Learning with scikit-learn[M].Packt Publishing Ltd.July 2017.Second published. 

In the actual application of the regression model, [scikit-learn(Sklearn)](https://scikit-learn.org/stable/index.html) library is mainly used. Corresponding changes are made to some descriptions. Independent variables are explanatory variables, which are features or attributes; its value is the eigenvectors. X is usually used to represent an eigenvector array or matrix. The dependent variable is the response variable, whose value is usually represented by Y(is the target value). The algorithms and models of machine learning are known as estimators, although the terms algorithm, model, and estimator are sometimes used confusedly.

The public health data is first to read in DataFrame format, but the Sklearn data processing is dominated by Numpy arrays and requires a conversion between them.

> Matrices are usually represented by uppercase letters(array with dimensions greater than or equal to 2), and vectors are represented by lowercase letters.


```python
import pandas as pd
import geopandas as gpd
import util

dataFp_dic={
    "ublic_Health_Statistics_byCommunityArea_fp":r'./data/Public_Health_Statistics-_Selected_public_health_indicators_by_Chicago_community_area.csv',
    "Boundaries_Community_Areas_current":r'./data/geoData/Boundaries_Community_Areas_current.shp',    
}

pubicHealth_Statistic=pd.read_csv(dataFp_dic["ublic_Health_Statistics_byCommunityArea_fp"])
community_area=gpd.read_file(dataFp_dic["Boundaries_Community_Areas_current"])
community_area.area_numbe=community_area.area_numbe.astype('int64')
pubicHealth_gpd=community_area.merge(pubicHealth_Statistic,left_on='area_numbe', right_on='Community Area')

print(pubicHealth_gpd.head())
```

       area area_num_1  area_numbe  comarea  comarea_id        community  \
    0   0.0         35          35      0.0         0.0          DOUGLAS   
    1   0.0         36          36      0.0         0.0          OAKLAND   
    2   0.0         37          37      0.0         0.0      FULLER PARK   
    3   0.0         38          38      0.0         0.0  GRAND BOULEVARD   
    4   0.0         39          39      0.0         0.0          KENWOOD   
    
       perimeter    shape_area     shape_len  \
    0        0.0  4.600462e+07  31027.054510   
    1        0.0  1.691396e+07  19565.506153   
    2        0.0  1.991670e+07  25339.089750   
    3        0.0  4.849250e+07  28196.837157   
    4        0.0  2.907174e+07  23325.167906   
    
                                                geometry  ...  \
    0  POLYGON ((-87.60914 41.84469, -87.60915 41.844...  ...   
    1  POLYGON ((-87.59215 41.81693, -87.59231 41.816...  ...   
    2  POLYGON ((-87.62880 41.80189, -87.62879 41.801...  ...   
    3  POLYGON ((-87.60671 41.81681, -87.60670 41.816...  ...   
    4  POLYGON ((-87.59215 41.81693, -87.59215 41.816...  ...   
    
       Childhood Lead Poisoning Gonorrhea in Females  Gonorrhea in Males  \
    0                       0.0               1063.3               727.4   
    1                       0.3               1655.4              1629.3   
    2                       2.5               1061.9              1556.4   
    3                       1.0               1454.6                1680   
    4                       0.4                610.2               549.1   
    
       Tuberculosis  Below Poverty Level  Crowded Housing  Dependency  \
    0           4.2                 26.1              1.6        31.0   
    1           6.7                 38.1              3.5        40.5   
    2           0.0                 55.5              4.5        38.2   
    3          13.2                 28.3              2.7        41.7   
    4           0.0                 23.1              2.3        34.2   
    
       No High School Diploma  Per Capita Income  Unemployment  
    0                    16.9              23098          16.7  
    1                    17.6              19312          26.6  
    2                    33.7               9016          40.0  
    3                    19.4              22056          20.6  
    4                    10.8              37519          11.0  
    
    [5 rows x 39 columns]
    

### 1.1 Simple linear regression of public health data
A correlation analysis is needed to determine whether the explanatory variables and the response variables are correlated before regression. We are returning to the 'Correlation analysis of public health data' to look at the results that have been calculated. 'Per Capita Income' and 'Childhood Blood Lead Level Screening' were selected from economic condition data and disease data, the correlation coefficient was -0.64, showing a negative linear correlation. In the regression session, Ordinary Lease Squares(OLS) or linear least squares are used to solve a simple linear regression equation, namely by minimizing the residual sum of squares to differentiate the regression coefficient, and make the results 0, get the value of the regression coefficient by solving equations set, and to construct a simple linear regression equation, or a model, estimator. If the response variables predicted by the model are all close to the observed value, then the model is fitted. The method to measure the fitting of the model by the residual sum of squares is called the residual sum of squares(RSS) cost function, also known as the *loss function*, which is used to define and measure the error of a model. The formula is the same as the residual sum of squares:$SS_{res} = \sum_{i=1}^n ( y_{i}- \widehat{ y_{i} } )^{2} $或$SS_{res} = \sum_{i=1}^n ( y_{i}- f( x_{i} ) )^{2} $, $ y_{i}$ is the observed value, $ \widehat{ y_{i}}$ and $f( x_{i})$ are the predicted value(regression value), $f(x_{i})$ is the estimator calculated. Therefore, differentiating the regression coefficient of the residual sum of squares is the process of finding the parameter value of the model by seeking the minimum value of the loss function, and the two are just different in expression.

In *Cavin Hackeling. Mastering Machine Learning with scikit-learn*, the author presents another method for solving simple linear regression coefficient, with the formula for calculating slope as follows: $\beta = \frac{cov(x,y)}{var(x)} = \frac{ \sum_{i=1}^n ( x_{i}- \overline{x} )(y_{i} - \overline{y} ) }{n-1} / \frac{ \sum_{i=1}^n ( x_{i}- \overline{x} )^{2} }{n-1}= \frac{ \sum_{i=1}^n ( x_{i}- \overline{x} )(y_{i} - \overline{y} ) }{( x_{i}- \overline{x} )^{2} }$，$var(x)$ is the variance of explanatory variable, $n$ is the total amount of training data, cov(x,y) is the covariance, $x_{i}$ represents the value of the $x_{i}$ $x$ in the training dataset, $\overline{x} $ is the mean of the explanatory variable, $y_{i}$ represents the $i$ $y$, $\overline{y}$ is the mean of the response variable. After the $\beta$ is obtained, the $\alpha $ intercept is obtained by the formula $\alpha = \overline{y} - \beta \overline{x} $. The calculation results are consistent with the `LinearRegression`(OLS) method provided by Sklearn. 

>  Variance is used to measure the degree of deviation of a set of values, usually marked as $ \sigma ^{2} $, that is, the square root of the standard deviation or the square root of the variance is the standard deviation. When all values in the set are equal, the variance is 0. The variance describes the distance of a random variable from its expected value, which is usually the mean of values.

> Covariance is used to measure the joint variation of two variables. Variance is a special case of covariance, that is, the covariance of a variable with itself. The covariance represents the error of the population of two variables, which is different from the variance of the error of only one variable. If the variation trend of the two variables is consistent, one is larger than its expected value, and the other is also larger than its expected value. The covariance between the two variables is positive. If the two variables' variation trend is the opposite, one is larger than its expected value, while the other is smaller than its expected value. The covariance between the two variables is negative. As all the values of the variable approach their expected values, the covariance approaches 0. When the two variables are statistically independent, the covariance is 0. But if the covariance is 0, it is not just a case of statistical independence; if there is no linear relationship between two variables, the covariance between them is 0. It is linearly independent but not necessarily relatively independent.

* Training (data)set, testing(data)set, and validation(data)set

Dataset is usually divided into a training dataset and a test dataset before the training model, and the validation dataset is often added. The training dataset is used to train the model, and the test dataset is used to evaluate the model performance according to the measurement criteria. The test dataset cannot contain the data in the training dataset; otherwise, it is difficult to assess whether the algorithm has really learned the generalization ability from the training dataset, or whether it simply remembers the training samples. A well-generalized model can effectively predict new data. If the model simply remembers the relationship between the explanatory variables and the response variables in the training dataset, it is called overfitting. Overfitting is usually treated by regularization. The validation dataset is often used to fine-tune the hyperparameters variables, which are usually configured manually, to control how the algorithm learns from the training data. There is no fixed proportion for each part division; usually, the training dataset accounts for 50%$ \sim $75%, the test dataset accounts for  10%$ \sim $25%, and the rest is the verification dataset.

The determination coefficient was calculated as 0.475165, and the score did not exceed 0.5. The simple linear regression model trained could not well predict the response variable based on the explanatory variable. Simple linear regression models are rarely used in the real world, and the complexity of the data makes us need to resort to more advantageous algorithms to solve practical problems. But the elaboration of simple linear regression step by step calculation method can give us a clearer understanding of the regression model. 


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ax1=pubicHealth_Statistic.plot.scatter(x='Per Capita Income',y='Childhood Blood Lead Level Screening',c='DarkBlue',figsize=(8,8),label='ground truth')
data_IncomeLead=pubicHealth_Statistic[['Per Capita Income','Childhood Blood Lead Level Screening']].dropna() #Sklearn solves models by removing null values from the dataset.
X=data_IncomeLead['Per Capita Income'].to_numpy().reshape(-1,1) #Convert the eigenvalue to the eigenvector matrix of Numpy
y=data_IncomeLead['Childhood Blood Lead Level Screening'].to_numpy() #Convert the target value to a vector in Numpy format

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LinearRegression
#Construct and fit the model
LR=LinearRegression().fit(X_train,y_train) #
#Model parameters
print("_"*50)
print("Sklearn slop:%.6f,intercept:%.6f"%(LR.coef_, LR.intercept_))
ax1.plot(X_train,LR.predict(X_train),'o-',label='linear regression',color='r',markersize=4)
ax1.set(xlabel='Per Capita Income',ylabel='Childhood Blood Lead Level Screening')

# The regression coefficient is calculated step by step.
income_variance=np.cov(X_train.T) #Use np.cov() to find the equation and the covariance. Notice the return value. If you find the covariance of two variables, the return value is the variance of each variable and the two variables' covariance matrix.
income_lead_covariance=np.cov(X_train.T,y_train)
print("income_variance=%.6f,income_lead_covariance=%.6f"%(income_variance,income_lead_covariance[0,1]))
beta=income_lead_covariance[0,1]/income_variance
alpha=y_train.mean()-beta*X_train.T.mean()
print("beta=%.6f,alpha=%.6f"%(beta,alpha))

ax1.plot(X.T.mean(),y.mean(),'x',label='X|y mean',color='green',markersize=20)
ax1.legend(loc='upper right', frameon=False)
plt.show()

#Simple linear regression equation - Regression Significance Test(regression coefficient test)
print("coefficient of determination，r_squared=%.6f"%LR.score(X_test,y_test)) 
```

    __________________________________________________
    Sklearn slop:-0.004310,intercept:499.029224
    income_variance=266830859.633061,income_lead_covariance=-1150133.371184
    beta=-0.004310,alpha=499.029224
    


<a href=""><img src="./imgs/8_1.png" height="auto" width="auto" title="caDesign"></a>


    coefficient of determination，r_squared=0.475165
    

### 1.2 K-nearest neighbor for public health data（k-nearest neighbors algorithm,k-NN）
#### 1.2.1 K-NN
K-NN is a simple model for regression and classification tasks. In 2 and 3 dimensional spaces, we can map explanatory variables and response variables to observable spaces. The eigenspace that defines the distance between all the members of the dataset is the metric space. However, if the explanatory number is large, it isn't easy to map variables to the observable 2 and 3-dimensional space but higher spatial dimensions. The members of all variables can be represented in this metric space. The principle of k-NN is to find adjacent samples. The classification task is to look at the categories of response variables corresponding to these samples that occupy the majority. In the regression task, the mean value of the response variables of adjacent samples is calculated. There is parameter k that needs to be manually controlled in k-NN; that is, the hyper-parameter k is used to specify the nearest neighbors' number. The determination coefficient of the model trained by different k values is different. In the following code, a k value interval is defined. By cycling k value, calculate and compare the determinant coefficient to find the k value at the maximum determination coefficient. The calculation result was that when k=5, the determination coefficient r_squared=0.675250. The regression model obtained using the k-NN algorithm had significantly improved predictive power compared to the simple linear regression model, indicating that the variance of the 'Childhood Blood Lead Level Screening' could be explained by the model in a large proportion. 

The distance between members in K-NN(or understood as the distance between two points or multiple points) is usually Euclidean Distance, and for n-dimensional space, $x_{1} (x_{11},x_{12}, \ldots ,x_{1n})$ and $x_{2} (x_{21},x_{22}, \ldots ,x_{2n})$ the formula of Euclidean Distance between two points is$d(x,y):= \sqrt{ ( x_{1} - \ y_{1} )^{2} +( x_{2} - \ y_{2} )^{2}+ \ldots +( x_{n} - \ y_{n} )^{2}  } = \sqrt{ \sum_{i=1}^n  ( x_{i} - \ y_{i} )^{2} } $，The Euclidean Distance formula in 2-dimensional space is the Pythagorean theorem of a right triangle:$\rho = \sqrt{ ( x_{2} - x_{1} )^{2} +(y_{2} - y_{1} )^{2} } $。

At the same time, Sklearn provides the weight parameter, which is especially useful for analyzing some variables affected by urban space's geographical location. The PySAL library has many methods that can be directly applied in the space weight section. In the weight parameter configuration, it contains three optional parameters: 'unifom', the weight of all points in each neighborhood is equal; 'distance', the weight is the distance reciprocal, that is, the more close to the query point, the neighbor point weight is higher, the farther is lower. 'callable', the user defines the weight. As the model between per capita income and children's blood lead level examination was established, the parameters were configured in the default, namely 'uniform' mode, because it did not involve analyzing the influence of spatial and geographical location.


```python
#Use the Sklearn library to train k-NN
from sklearn.neighbors import KNeighborsRegressor
import math
fig, axs=plt.subplots(1,2,figsize=(25,11))

k_neighbors=range(1,15,1)
r_squared_temp=0
for k in k_neighbors:
    knn=KNeighborsRegressor (n_neighbors=k)
    knn.fit(X_train,y_train)
    if r_squared_temp<knn.score(X_test,y_test): #knn-regression significance test(regression coefficient test)
        r_squared_temp=knn.score(X_test,y_test) 
        k_temp=k
knn=KNeighborsRegressor (n_neighbors=k_temp).fit(X_train,y_train)
print("In the interval %s, the largest r_squared=%.6f, the corresponding k=%d"%(k_neighbors,knn.score(X_test,y_test) ,k_temp))

pubicHealth_Statistic.plot.scatter(x='Per Capita Income',y='Childhood Blood Lead Level Screening',c='DarkBlue',label='ground truth',ax=axs[0])
X_train_sort=np.sort(X_train,axis=0)
axs[0].plot(X_train_sort,knn.predict(X_train_sort),'o-',label='knn regressionn',color='r',markersize=4)

axs[0].set(xlabel='Per Capita Income',ylabel='Childhood Blood Lead Level Screening')
axs[0].plot(X.T.mean(),y.mean(),'x',label='X|y mean',color='green',markersize=20)

#About K-NN
Xy=data_IncomeLead.to_numpy()
#A - A custom function that returns a k-nearest neighbor index
def k_neighbors_entire(xy,k=3):
    import numpy as np
    '''
    function - Returns the nearest point coordinates for a specified number of neighbors
    
    Paras:
    xy - A 2-dimensional array of point coordinates, for example
        array([[23714. ,   364.7],
              [21375. ,   331.4],
              [32355. ,   353.7],
              [35503. ,   273.3]]
    k - Specify the number of neighbors
    
    return:
    neighbors - Returns the index of each point, and all the index of adjacent points specified number for each point
    '''
    neighbors=[(s,np.sqrt(np.sum((xy-xy[s])**2,axis=1)).argsort()[1:k+1]) for s in range(xy.shape[0])]
    return neighbors
    
neighbors=k_neighbors_entire(Xy,k=5)
any_pt_idx=70
any_pt=np.take(Xy,neighbors[any_pt_idx][0],axis=0)
neighbor_pts=np.take(Xy,neighbors[any_pt_idx][1],axis=0)

axs[0].plot(any_pt[0],any_pt[1],'x',label='any_pt',color='black',markersize=10)
axs[0].scatter(neighbor_pts[:,0],neighbor_pts[:,1],c='orange',label='neighbors')

#B - pointpats under the PySAL library are used to find the nearest neighbor points
from pointpats import PointPattern
pubicHealth_Statistic.plot.scatter(x='Per Capita Income',y='Childhood Blood Lead Level Screening',c='DarkBlue',label='ground truth',ax=axs[1])
Xy=data_IncomeLead.to_numpy()
pp=PointPattern(Xy)
pp_neighbor=pp.knn(5)

any_pt=np.take(Xy,any_pt_idx,axis=0)
neighbor_pts_idx=np.take(pp_neighbor[0],any_pt_idx,axis=0)
neighbor_pts=np.take(Xy,neighbor_pts_idx,axis=0)

axs[1].plot(any_pt[0],any_pt[1],'x',label='any_pt',color='black',markersize=10)
axs[1].scatter(neighbor_pts[:,0],neighbor_pts[:,1],c='orange',label='neighbors')
for coodi in neighbor_pts:
    axs[1].arrow(any_pt[0],any_pt[1],coodi[0]-any_pt[0], coodi[1]-any_pt[1], head_width=1, head_length=1,color="gray",linestyle="--" ,length_includes_head=True)
    
from matplotlib.text import OffsetFrom
axs[1].annotate("y_mean=%s"%neighbor_pts[:,1].mean(), xy=any_pt, xycoords="data",xytext=(50000, 250), va="top", ha="center",bbox=dict(boxstyle="round", fc="w"),arrowprops=dict(arrowstyle="->"))    

axs[0].legend(loc='upper right', frameon=False)
axs[1].legend(loc='upper right', frameon=False)
plt.show()
```

    In the interval (1, 15),the largest r_squared=0.675250,the corresponding k=5
    


<a href=""><img src="./imgs/8_2.png" height="auto" width="auto" title="caDesign"></a>


#### 1.2.2 mean absolute error(MAE) and mean squared error(MSE)

Mean absolute error(MAE), also known as L1 norm loss(L1_Loss), is the mean value of the absolute error of prediction results. It is a regression loss function, and its formula is:$MAE(y, \widehat{y} )= \frac{1}{ n_{samples} }  \sum_{i=0}^{ n_{samples}-1 }  |   y_{i} - \widehat{ y_{i} } |  $，$ \widehat{ y_{i}}$ is the predicted value, $ y_{i} $ is the observed value, $[0, \infty]$ is the scope. The L1 norm loss function MAE can better measure the regression model's quality, but the existence of absolute value leads to the function is not smooth. Although there is a stable gradient for all input values, which will not lead to a gradient explosion problem and has a relatively robust solution, it is a folding point at the center, which cannot be differentiated and is not convenient to solve.


The mean square error(MSE) is a more commonly used index than MAE, also known as L2_norm loss(L2_Loss is the most commonly used loss function), is the residual sum of squares($SS_{res} $) described in the regression section and then divided by the sample size, and the formula is:$MSE(y, \widehat{y} )= \frac{1}{ n_{samples} }  \sum_{i=0}^{ n_{samples}-1 }  (y_{i} - \widehat{ y_{i} } )^{2}$。L2 norm loss MSE, each point continuous smooth, easy to differentiate, has a relatively stable solution. However, it is not particularly robust. When the input value of the function is far away from the central value, the gradient is very large when the gradient descent method is used to solve the problem, which results in the substantial updating of the network weight during the neural network training process and makes the network unstable. In extreme cases, the weight value may overflow, namely, gradient explosion.

From the following graph, it is easy to observe the function changes by printing the function graph of MAE and MSE. In the regression part, we differentiate the residual sum of squares to a and b and set the result equal to 0 to obtain the regression equation. The loss function is solved to make it easier to understand what a loss function is extended to Sklearn machine learning. Note that the following code creates a simple linear regression equation:$y= ax$, with no intercept of $b$ to simplify the operation.


> The norm is a function that has the concept of 'length'. In linear algebra, functional analysis, and related mathematics, it is a function that assigns a nonzero positive length or magnitude to all vectors in a vector space. For example, a 2-dimensional Euclidean geometric space $R^{2} $ has a Euclidean norm. An element  $(x_{i} ,y_{i})$ in this vector space is often drawn in the Cartesian coordinate system as an arrow starting at the origin, and the Euclidean norm of each vector is the length of the arrow. Or you can think of it as a vector space(namely metric space) that has a magnitude, and the way to measure that magnitude is to measure it in terms of norms, and different norms can measure that magnitude. Its norm is defined as the norm of a vector is a function $||x||$, which satisfies non-negative $||x||>=0$, homogeneity $||cx||=| c| ||x||$, triangle inequality $||x+y||<=||x||+||y||$.

> When describing k-NN, talking about metric space, for 2,3-dimensional vectors, geometric figures can be drawn. Still, for multidimensional, high dimensional, beyond three-dimensional space, the concept of the norm is introduced, so that the distance between two vectors of any dimension can be defined. The norm of commonly used vectors are:
> 1. L1 norm:$||x||$ is the sum of the absolute values of each element of the $x$ vector, and the formula is: $||x||_{1} = \sum_i | x_{i} | = |x_{1} |+| x_{2} |+ \ldots +| x_{i} |$
2. L2 norm:$||x||$ is the 1/2 power of the sum of squares of each element of the $x$ vector, and the formula is: $||x||_{2} =  \sqrt{(\sum_i   x_{i} ^{2} })  =   \sqrt{x_{1}^{2} +x_{2}^{2}  + \ldots + x_{i}^{2}  } $
3. Lp norm:$||x||$ is the absolute value p power of each element of $x$ vector to the 1/p power, the formula is:$||x||_{p} =   ( \sum_i  | x_{i} |^{p}  )  ^{1/p} $
4. $L^{ \circ } $ norm: $||x||$ is the absolute value of the maximum element in all elements absolution of $x$ vector.


```python
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("coefficient of determination=%s"%r2_score(y_test,knn.predict(X_test)))
print("mean absolute error=%s"%mean_absolute_error(y_test,knn.predict(X_test)))
print("mean squared error=%s"%mean_squared_error(y_test,knn.predict(X_test)))

# Print the MAE function graph.
fig, axs=plt.subplots(1,2,figsize=(18,8))

import sympy
import pandas as pd
a=sympy.symbols('a')
data_IncomeLead_copy=data_IncomeLead.copy(deep=True)
data_IncomeLead_copy.rename(columns={"Per Capita Income":"PCI","Childhood Blood Lead Level Screening":"CBLLS"},inplace=True)

data_IncomeLead_copy["residual"]=data_IncomeLead_copy.apply(lambda row:row.CBLLS-(a*row.PCI),axis=1)
data_IncomeLead_copy["abs_residual"]=data_IncomeLead_copy.residual.apply(lambda row:abs(row))
n_s=data_IncomeLead_copy.shape[0]
MAE=data_IncomeLead_copy.abs_residual.sum()/n_s
MAE_=sympy.lambdify([a],MAE,"numpy")
a_val=np.arange(-100,100,1) #Assumed a

axs[0].plot(a_val,MAE_(a_val),'-',label='MAE function')

#Print the MSE function graph.
data_IncomeLead_copy["residual_squared"]=data_IncomeLead_copy.residual.apply(lambda row:row**2)
MSE=data_IncomeLead_copy.residual_squared.sum()/n_s
MSE_=sympy.lambdify([a],MSE,"numpy")
axs[1].plot(a_val,MSE_(a_val),'-',label='MSE function')

axs[0].legend(loc='upper right', frameon=False)
axs[1].legend(loc='upper right', frameon=False)
plt.show()
```

    coefficient of determination=0.675249990123995
    mean absolute error=59.75384615384613
    mean squared error=4912.952276923076
    


<a href=""><img src="./imgs/8_3.png" height="auto" width="auto" title="caDesign"></a>


#### 1.2.3 Scaling of eigenvalues(standardization)
The feature value of per capita income range in [ 9016. 87163.] with us dollars unit. However, the feature value of children's blood level test range in [133.6 605.9], and the unit is per 1000 people. The two feature units are different, and the value range is also very different, which will inevitably affect the model training. If its range is scaled to the same value range, the learning algorithm will perform better. The scaling of the feature value can use the `sklearn.preprocessing.StandardScaler` method provided by Sklearn library, the calculation formula is: $z=(x- \mu )/s$, $\mu $ is the mean of the training sample, $s$ is the standard deviation fo the training sample.


The calculation results show that the determination coefficient after the standardization of the feature value is 0.72, which is greater than the value 0.68 before the feature value processing, and the predicted ability of the model can be improved again.


```python
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_train_scaled=SS.fit_transform(X_train) 
X_test_scaled=SS.fit_transform(X_test) 

k_neighbors=range(1,15,1)
r_squared_temp=0
for k in k_neighbors:
    knn_=KNeighborsRegressor (n_neighbors=k)
    knn_.fit(X_train_scaled,y_train)
    if r_squared_temp<knn_.score(X_test_scaled,y_test): #knn-regression significance test(regression coefficient test)
        r_squared_temp=knn_.score(X_test_scaled,y_test) 
        k_temp=k
knn_=KNeighborsRegressor(n_neighbors=k_temp).fit(X_train_scaled,y_train)

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("coefficient of determination=%s"%r2_score(y_test,knn_.predict(X_test_scaled)))
print("mean absolute error=%s"%mean_absolute_error(y_test,knn_.predict(X_test_scaled)))
print("mean squared error=%s"%mean_squared_error(y_test,knn_.predict(X_test_scaled)))

from numpy.polynomial import polyutils as pu
Per_Capita_Income_domain=pu.getdomain(X.reshape(-1))
Childhood_Blood_Lead_Level_Screening_domain=pu.getdomain(y)
print("Per Capita Income domain=%s"%Per_Capita_Income_domain)
print("Childhood Blood Lead Level Screening domain=%s"%Childhood_Blood_Lead_Level_Screening_domain)
```

    coefficient of determination=0.7150373630356484
    mean absolute error=56.32587412587413
    mean squared error=4311.032466624287
    Per Capita Income domain=[ 9016. 87163.]
    Childhood Blood Lead Level Screening domain=[133.6 605.9]
    

### 1.3 Multiple regression of public health data
#### 1.3.1 Multiple linear regression
There are still many ways to print a chart to observe data association with more than three multiple explanatory variables (multidimensional or high-dimensional arrays), such as using a parallel coordinates plot. The response variable is 'Childhood Blood Lead Level Screening'; others are explanatory variables. The trend of the polylines can be used to initially determine whether the explanatory variables are positively or negatively correlated with the response variables. In contrast, the distribution density of the polylines indicates the numerical distribution of each variable.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import plotly.express as px

columns=['Per Capita Income','Below Poverty Level','Crowded Housing','Dependency','No High School Diploma','Unemployment','Childhood Blood Lead Level Screening']
data_Income=pubicHealth_Statistic[columns].dropna() #Sklearn solves models by removing null values from the dataset.

fig = px.parallel_coordinates(data_Income, labels=columns,color_continuous_scale=px.colors.diverging.Tealrose,color_continuous_midpoint=2,color='Childhood Blood Lead Level Screening')
fig.show()
```

<a href=""><img src="./imgs/8_4.png" height="auto" width="auto" title="caDesign"></a>


The determination coefficient is 0.48 by training the regression model with one feature 'Per Capita Income'; the determination coefficient is 0.76 with six explanatory variables, multiple features. The increased explanatory variables improve the performance of the model compared with the single feature. At the same time, k-NN is compared, when k=4, the six features are used as explanatory variables, and the determination coefficient is 0.82, while one feature is 0.72. The model's performance is also greatly improved, indicating that the increased explanatory variables contribute to the model's prediction.


```python
X=data_Income[['Per Capita Income','Below Poverty Level','Crowded Housing','Dependency','No High School Diploma','Unemployment']].to_numpy() #Convert the feature values to the eigenvector matrix of Numpy
y=data_Income['Childhood Blood Lead Level Screening'].to_numpy() #Convert the target value to a vector in Numpy format

SS=StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train_scaled=SS.fit_transform(X_train) 
X_test_scaled=SS.fit_transform(X_test)

LR_m=LinearRegression()
LR_m.fit(X_train_scaled, y_train)
print("Linear Regression - Accuracy on test data: {:.2f}".format(LR_m.score(X_test_scaled, y_test)))
y_pred=LR_m.predict(X_train_scaled)


# Use k-NN
from sklearn.neighbors import KNeighborsRegressor
k_neighbors=range(1,15,1)
r_squared_temp=0
for k in k_neighbors:
    knn=KNeighborsRegressor (n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    if r_squared_temp<knn.score(X_test_scaled,y_test): #knn-Regression significance test(regression coefficient test)
        r_squared_temp=knn.score(X_test_scaled,y_test) 
        k_temp=k
knn=KNeighborsRegressor (n_neighbors=k_temp).fit(X_train_scaled,y_train)
print("K-NN in the interval %s,the largest r_squared=%.6f,the corresponding k=%d"%(k_neighbors,knn.score(X_test_scaled,y_test) ,k_temp))
```

    Linear Regression - Accuracy on test data: 0.76
    K-NN in the interval range(1, 15),the largest r_squared=0.815978,the corresponding k=4
    

#### 1.3.2 Polynomial regression
Many causal relationships are not linear in the real world, and multiple approaches can be used to model the nonlinear relationship between explanatory variables and response variables. Here, the polynomial regression method is used to fit the model's curve in the differential part, and the results obtained are much better than those obtained in the linear model. Firstly, the scatter plots of the response variable 'Childhood Blood Lead Level Screening' and all other explanatory variables of economic conditions are printed. It is observed that the feature of 'Per Capita Income' tends to be linear, while other features seem to show a certain radian. 


```python
import plotly.express as px
fig=px.scatter_matrix(data_Income)

fig.update_layout(
    autosize=True,
    width=1800,
    height=1800,
    )
fig.show()
```

<a href=""><img src="./imgs/8_5.png" height="auto" width="auto" title="caDesign"></a>

Only the explanatory variable 'Below Poverty Level' is selected To facilitate the observation of the fitting curves, and the response variable is still 'Childhood Blood Lead Level Screening'. Incoming parameters  'degree' of `PolynomialFeatures` method configure the degree of the feature polynomial; the default value is 2, different cycle degrees value at the same time, draw the corresponding curve fitting and obtain the degree value at the highest value by comparing the value of the determination coefficient. When the degree is 4, the determination coefficient value of 0.81 is the highest.


```python
X=data_Income['Below Poverty Level'].to_numpy().reshape(-1,1) #Convert the feature values to the eigenvector matrix of Numpy
y=data_Income['Childhood Blood Lead Level Screening'].to_numpy() #Convert the target value to a vector in Numpy format

def PolynomialFeatures_regularization(X,y,regularization='linear'):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler    
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    '''
    function - Polynomial regression degree selection and regularization
    
    X - Explanatory variable
    y - Response variable
    regularization - The regularization methods are 'Ridge' and 'LASSO', when 'linear', is not regularized.
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    SS=StandardScaler()
    X_train_scaled=SS.fit_transform(X_train) 
    X_test_scaled=SS.fit_transform(X_test)

    degrees=np.arange(1,16,1)
    fig_row=3
    fig_col=degrees.shape[0]//fig_row
    fig, axs=plt.subplots(fig_row,fig_col,figsize=(21,12))
    r_squared_temp=0
    p=[(r,c) for r in range(fig_row) for c in range(fig_col)]
    i=0
    for d in degrees:
        if regularization=='linear':
            model=Pipeline([('poly', PolynomialFeatures(degree=d)),
                            ('regular', LinearRegression(fit_intercept=False))])
        elif regularization=='Ridge':
            model=Pipeline([('poly', PolynomialFeatures(degree=d)),
                            ('regular', Ridge())])            
        elif regularization=='LASSO':
            model=Pipeline([('poly', PolynomialFeatures(degree=d)),
                            ('regular', Lasso())])             
        
        reg=model.fit(X_train_scaled,y_train)
        x_=X_train_scaled.reshape(-1)
        print("The training dataset-r_squared=%.6f,the test dataset-r_squared=%.6f,the corresponding degree=%d"%(reg.score(X_train_scaled,y_train),reg.score(X_test_scaled,y_test) ,d))  
        print("Coefficient:",reg['regular'].coef_)
            
        
        print("_"*50)
        
        X_train_scaled_sort=np.sort(X_train_scaled,axis=0)

        axs[p[i][0]][p[i][1]].scatter(X_train_scaled.reshape(-1),y_train,c='black')
        axs[p[i][0]][p[i][1]].plot(X_train_scaled_sort.reshape(-1),reg.predict(X_train_scaled_sort),label='degree=%s'%d)
        axs[p[i][0]][p[i][1]].legend(loc='lower right', frameon=False)


        if r_squared_temp<reg.score(X_test_scaled,y_test): #knn-Regression significance test(regression coefficient test)
            r_squared_temp=reg.score(X_test_scaled,y_test) 
            d_temp=d   
        i+=1    

    plt.show()        
    model=Pipeline([('poly', PolynomialFeatures(degree=d_temp)),
                    ('linear', LinearRegression(fit_intercept=False))])
    reg=model.fit(X_train_scaled,y_train)    
    print("_"*50)
    print("K-NN in the interval %s,the largest r_squared=%.6f,the corresponding degree=%d"%(degrees,reg.score(X_test_scaled,y_test) ,d_temp))  
    
    return reg
reg=PolynomialFeatures_regularization(X,y,regularization='linear')    
```

    "The training dataset-r_squared=0.383777,the test dataset-r_squared=0.649537,the corresponding degree=1
    Coefficient: [387.116       72.44793532]
    __________________________________________________
    "The training dataset-r_squared=0.525068,the test dataset-r_squared=0.799680,the corresponding degree=2
    Coefficient: [415.15064819 106.59487426 -28.03464819]
    __________________________________________________
    "The training dataset-r_squared=0.533993,the test dataset-r_squared=0.812620,the corresponding degree=3
    Coefficient: [421.5556279   97.40720189 -40.42263154   4.91204797]
    __________________________________________________
    "The training dataset-r_squared=0.534236,the test dataset-r_squared=0.812928,the corresponding degree=4
    Coefficient: [422.04661766 100.89409119 -41.34031714   2.74722153   0.61986186]
    __________________________________________________
    "The training dataset-r_squared=0.552199,the test dataset-r_squared=0.782880,the corresponding degree=5
    Coefficient: [406.48814993 105.12024109  21.14787779  -4.00537944 -22.95458517
       5.78979431]
    __________________________________________________
    "The training dataset-r_squared=0.560669,the test dataset-r_squared=0.736790,the corresponding degree=6
    Coefficient: [404.61456156  70.62356274  25.41030875  59.54880252 -30.36086237
     -13.31892642   4.58301824]
    __________________________________________________
    "The training dataset-r_squared=0.561443,the test dataset-r_squared=0.721949,the corresponding degree=7
    Coefficient: [407.30609959  71.55476046   5.22112238  57.88741266  -8.34375345
     -15.14286132  -1.01076346   1.27306864]
    __________________________________________________
    "The training dataset-r_squared=0.561457,the test dataset-r_squared=0.720343,the corresponding degree=8
    Coefficient: [ 4.07432749e+02  6.99138411e+01  3.76518801e+00  6.27705806e+01
     -6.78783742e+00 -1.87285260e+01 -1.02408692e+00  2.05015447e+00
     -1.55136817e-01]
    __________________________________________________
    "The training dataset-r_squared=0.561509,the test dataset-r_squared=0.729657,the corresponding degree=9
    Coefficient: [406.45186965  68.10329352  15.11566172  70.56903136 -27.35899397
     -24.53485275  10.65409882   2.10527231  -2.31237115   0.40977271]
    __________________________________________________
    "The training dataset-r_squared=0.584794,the test dataset-r_squared=0.515215,the corresponding degree=10
    Coefficient: [ 416.9565451   -22.79512573 -161.63613274  587.51387394  385.40821889
     -779.51922596 -228.82928612  393.70946989    3.93889364  -67.9693672
       12.58496909]
    __________________________________________________
    "The training dataset-r_squared=0.584989,the test dataset-r_squared=0.417839,the corresponding degree=11
    Coefficient: [ 418.48831932  -17.40245655 -189.24329768  547.48572931  475.2130813
     -712.55205508 -333.34006228  361.02984501   52.9867451   -67.1027363
        4.49118821    1.46137896]
    __________________________________________________
    "The training dataset-r_squared=0.585052,the test dataset-r_squared=0.382826,the corresponding degree=12
    Coefficient: [ 417.72971214  -14.7899818  -173.36519353  525.70034608  418.25351445
     -658.55342189 -265.56473888  304.02297799   24.32398421  -40.78607186
        6.51356957   -2.88642496    0.71844643]
    __________________________________________________
    "The training dataset-r_squared=0.590331,the test dataset-r_squared=-0.114724,the corresponding degree=13
    Coefficient: [  424.34973114    42.62084484  -308.78664676   -18.49939371
       970.06062603   653.99458513 -1197.12058689  -902.98494196
       805.19427462   392.78276665  -310.98336865   -35.52661013
        49.33246865    -7.44339216]
    __________________________________________________
    "The training dataset-r_squared=0.603513,the test dataset-r_squared=-7.930648,the corresponding degree=14
    Coefficient: [  409.97673431   135.20495819   146.41410195 -1025.5824304
     -1545.59989065  3716.12409702  3485.70387856 -5011.4854729
     -2794.26314994  3187.49880291   790.63782529  -988.43720983
        -6.87130152   120.06999987   -19.88476241]
    __________________________________________________
    "The training dataset-r_squared=0.604060,the test dataset-r_squared=-3.340680,the corresponding degree=15
    Coefficient: [ 4.11556610e+02  1.61553840e+02  8.85250922e+01 -1.38796632e+03
     -1.12016163e+03  5.09282051e+03  2.32199847e+03 -7.12596454e+03
     -1.28518027e+03  4.64340365e+03 -2.16517083e+02 -1.40585556e+03
      3.29106673e+02  1.41310367e+02 -6.37668300e+01  6.73200032e+00]
    __________________________________________________
    


<a href=""><img src="./imgs/8_6.png" height="auto" width="auto" title="caDesign"></a>


    __________________________________________________
    K-NN in the interval[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15],the largest r_squared=0.812928,the corresponding degree=4
    
 
When multiple features are input as explanatory variables, the maximum determination coefficient of polynomial regression is 0.76, and the degree is 1, that is, the linear model; this indicates that for this dataset, when the following six variables are used as explanatory variables, the linear regression model's prediction accuracy is higher than that of polynomial regression.

```python
X_=data_Income[['Per Capita Income','Below Poverty Level','Crowded Housing','Dependency','No High School Diploma','Unemployment',]].to_numpy() #Convert the feature values to the eigenvector matrix of Numpy
y_=data_Income['Childhood Blood Lead Level Screening'].to_numpy() #Convert the target value to a vector in Numpy format

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.33, random_state=42)
SS=StandardScaler()
X_train_scaled=SS.fit_transform(X_train) 
X_test_scaled=SS.fit_transform(X_test)

degrees=np.arange(1,16,1)
r_squared_temp=0
for d in degrees:
    model=Pipeline([('poly', PolynomialFeatures(degree=d)),
                    ('linear', LinearRegression(fit_intercept=False))])
    reg=model.fit(X_train_scaled,y_train)
    x_=X_train_scaled.reshape(-1)
    
    if r_squared_temp<reg.score(X_test_scaled,y_test): #knn-Regression significance test(regression coefficient test)
        r_squared_temp=reg.score(X_test_scaled,y_test) 
        d_temp=d   
     
model=Pipeline([('poly', PolynomialFeatures(degree=d_temp)),
                ('linear', LinearRegression(fit_intercept=False))])
reg=model.fit(X_train_scaled,y_train)                
print(""In the interval%s,the largest r_squared=%.6f,the corresponding degree=%d"%(degrees,reg.score(X_test_scaled,y_test) ,d_temp))    
```

    In the interval [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15],the largest r_squared=0.764838,the corresponding degree=1
    

#### 1.3.4 Regularization
In polynomial regression, the determination coefficient is calculated simultaneously on the training dataset and the test dataset. It can be observed from the results that when the degree increases, the determination coefficient on the training dataset increases, but the determination coefficient obtained on the test dataset does not keep consistent with the change on the training dataset; this problem is called overfitting because the value of a single parameter obtained may be very large, and when the model is faced with new data, it will have great fluctuation, which is a problem that the model contains huge variance error; this is since there are many features in the sample, but the sample size is small, so the model is prone to overfitting. This experiment's feature number is 6, and the sample size is 76, with relatively small sample size. One way to solve overfitting is to reduce the number of features; the other is regularization.

Regularization is a collection of techniques that can be used to prevent overfitting, and Sklearn provides methods ridge(L2 regularization), and LASSO regression(L1 regularization). For a multiple linear regression model: $y=  \beta _{0} + \beta _{1} x_{1} + \beta _{2} x_{2}+  \ldots + \beta _{n} x_{n} $，The predicted value of a sample $X^{(i)} $ is: $  \widehat{y} ^{(i)} =  \beta _{0} + \beta _{1}  X_{1}^{(i)}  + \beta _{2} X_{2}^{(i)}+  \ldots + \beta _{n} X_{n}^{(i)}$，The model ultimately requires solution parameters $\beta = (  \beta _{0}, \beta _{1}, \ldots ,\beta _{n})^{T} $，Make the mean square error(MSE as small as possible. However, through the above experiments, it is observed from the coefficients obtained that some coefficients are relatively large, and the predicted values of the test dataset may fluctuate greatly. Therefore, to improve the generalization ability of the model, the parameter $\beta$ is limited, Ridge regression modifies the RSS loss function by adding the L2 norm fo the coefficient. The formula is as follows: $RSS_{ridge}= \sum_{i=1}^n  ( y_{i} -  x_{i}^{T} \beta   )^{2} + \lambda \sum_{j=1}^p    \beta _{j} ^{2} $。

It can be found that the determination coefficient on the training dataset no longer has an obvious trend to increase with the increase to degree parameter and is relatively stable utilizing ridge regression regularization and printing calculation results and graphs. You can also see from the printed figure that the fitted curve is smoother than before regularization. The highest determination coefficient 0.812620, is unchanged, but the value of the degree changes from 4 to 3.


Simultaneously, the strength of the parameter 'alpha'  of Ridge can be modified to optimize further. The higher the value, the stronger the regularization. When the 'alpha' parameter is large enough that the almost regular term is in effect in the loss function, and the curve becomes a line parallel to the $x$ axis. The default value is 1.0.


```python
reg=PolynomialFeatures_regularization(X,y,regularization='Ridge')   
```

    The training dataset-r_squared=0.383629,the test dataset-r_squared=0.643434,the corresponding degree=1
    Coefficient: [ 0.         71.02738756]
    __________________________________________________
    The training dataset-r_squared=0.524482,the test dataset-r_squared=0.795372,the corresponding degree=2
    Coefficient: [  0.         103.02552946 -26.7958948 ]
    __________________________________________________
    The training dataset-r_squared=0.533428,the test dataset-r_squared=0.808643,the corresponding degree=3
    Coefficient:  [  0.          93.50225814 -39.86167095   5.18553358]
    __________________________________________________
    The training dataset-r_squared=0.532731,the test dataset-r_squared=0.808167,the corresponding degree=4
    Coefficient:  [  0.          90.14428324 -38.91589082   7.34685881  -0.62571603]
    __________________________________________________
    The training dataset-r_squared=0.550438,the test dataset-r_squared=0.788672,the corresponding degree=5
    Coefficient:  [  0.          93.84764775  12.87360517   1.64678343 -20.79054041
       4.98427692]
    __________________________________________________
    The training dataset-r_squared=0.559629,the test dataset-r_squared=0.760984,the corresponding degree=6
    Coefficient:  [  0.          71.60728566  16.54120019  49.61630718 -26.29291188
     -10.22653493   3.66477484]
    __________________________________________________
    The training dataset-r_squared=0.560749,the test dataset-r_squared=0.733622,the corresponding degree=7
    Coefficient:  [  0.          71.44945769   1.37430615  49.49306876  -3.83986106
     -12.38314706  -2.68896087   1.48586176]
    __________________________________________________
    The training dataset-r_squared=0.560380,the test dataset-r_squared=0.727878,the corresponding degree=8
    Coefficient:  [ 0.         72.51848056  2.84961265 38.43406208 -7.03936925 -0.11063406
     -2.99758766 -1.70728792  0.68818115]
    __________________________________________________
    The training dataset-r_squared=0.560436,the test dataset-r_squared=0.722722,the corresponding degree=9
    Coefficient:  [ 0.         71.74760112  2.42062079 37.28514741 -0.1208548   2.81988313
     -9.26801727 -1.90551868  2.18622176 -0.29603236]
    __________________________________________________
    The training dataset-r_squared=0.560700,the test dataset-r_squared=0.729012,the corresponding degree=10
    Coefficient: [  0.          72.63902932   3.38850893  37.02141463   0.43466143
      -0.72725328 -10.83295468   1.07563872   2.30663806  -0.96127219
       0.12701471]
    __________________________________________________
    The training dataset-r_squared=0.560759,the test dataset-r_squared=0.731137,the corresponding degree=11
    Coefficient:  [ 0.00000000e+00  7.25863298e+01  3.70857411e+00  3.73108505e+01
      3.23411623e-01 -5.67036249e-01 -1.16834151e+01  7.06644667e-01
      2.99160544e+00 -9.36936734e-01 -1.92812738e-02  2.77122165e-02]
    __________________________________________________
    The training dataset-r_squared=0.565228,the test dataset-r_squared=0.775583,the corresponding degree=12
    Coefficient:  [  0.          74.22746918   3.47231345  44.25822447   8.24104034
      -3.3859839   -6.69162561 -20.68883645  -6.92545254  16.24904417
       0.66709867  -3.60565928   0.68256426]
    __________________________________________________
    The training dataset-r_squared=0.565621,the test dataset-r_squared=0.775117,the corresponding degree=13
    Coefficient:  [  0.          72.93631205   5.27178629  44.61305599  10.57069606
       0.72257378  -7.76863364 -17.35067207 -16.14739195  10.97869673
       8.21623175  -3.13373255  -0.92815325   0.29528205]
    __________________________________________________
    The training dataset-r_squared=0.567803,the test dataset-r_squared=0.718550,the corresponding degree=14
    Coefficient: [  0.          72.48986814  -0.38802092  46.37996174   9.65490388
       3.80119261   1.83709905 -17.13541961  -4.31373318  -5.64511991
      -5.52665678  10.67428275   0.72497242  -2.72951948   0.52176813]
    __________________________________________________
    The training dataset-r_squared=0.569976,the test dataset-r_squared=0.725224,the corresponding degree=15
    Coefficient:  [  0.          72.4994701   -0.11084462  50.1640638   10.13680556
       5.66365663   1.07615047 -22.08021398  -5.86374799 -14.04837917
       1.66044719  18.7517073   -5.21311107  -3.84642138   1.85506701
      -0.21317021]
    __________________________________________________
    

    C:\Users\richi\anaconda3\envs\pdal\lib\site-packages\sklearn\linear_model\_ridge.py:148: LinAlgWarning:
    
    Ill-conditioned matrix (rcond=1.27167e-17): result may not be accurate.
    
    


<a href=""><img src="./imgs/8_7.png" height="auto" width="auto" title="caDesign"></a>


    __________________________________________________
    In the interval [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15],the largest r_squared=0.812620,the corresponding degree=3
    

LASSO（Least Absolute Shrinkage and Selection Operator Regression）algorithm can add L1 norm on the loss function to punish coefficient, the formula is: $RSS_{lasso}= \sum_{i=1}^n ( y_{i} - x_{i}^{T} \beta )^{2} + \lambda \sum_{j=1}^p  |\beta _{j}| $. LASSO's feature makes part of $\beta$ 0, which can be used as the feature selection. The feature with a coefficient of 0 indicates that the explanatory variable has little effect on improving model accuracy. Compared with Ridge, LASSO can be used for the dataset with more features or Ridge because Ridge is more accurate.


```python
from warnings import filterwarnings
filterwarnings('ignore')
reg_lasso=PolynomialFeatures_regularization(X,y,regularization='LASSO')   
```

    The training dataset-r_squared=0.383704,the test dataset-r_squared=0.645268,the corresponding degree=1
    Coefficient:  [ 0.         71.44793532]
    __________________________________________________
    The training dataset-r_squared=0.524848,the test dataset-r_squared=0.797057,the corresponding degree=2
    Coefficient: [  0.         104.49105524 -27.13097434]
    __________________________________________________
    The training dataset-r_squared=0.533732,the test dataset-r_squared=0.809695,the corresponding degree=3
    Coefficient:  [  0.          95.92468724 -38.68189204   4.58082344]
    __________________________________________________
    The training dataset-r_squared=0.533767,the test dataset-r_squared=0.809769,the corresponding degree=4
    Coefficient:  [ 0.00000000e+00  9.60447721e+01 -3.88303966e+01  4.49967222e+00
      3.38634247e-02]
    __________________________________________________
    The training dataset-r_squared=0.549919,the test dataset-r_squared=0.797657,the corresponding degree=5
    Coefficient:  [  0.          99.25564274   0.          -0.         -15.43948323
       3.85516722]
    __________________________________________________
    The training dataset-r_squared=0.550943,the test dataset-r_squared=0.798039,the corresponding degree=6
    Coefficient: [  0.         100.89815487   0.17286352   0.93205514 -15.80215342
       3.16762738   0.19793767]
    __________________________________________________
    The training dataset-r_squared=0.552514,the test dataset-r_squared=0.794852,the corresponding degree=7
    Coefficient: [ 0.00000000e+00  9.92102981e+01  1.88055289e+00  4.25201475e+00
     -1.56479603e+01  2.14764569e+00  3.47881123e-02  1.00395677e-01]
    __________________________________________________
    The training dataset-r_squared=0.553184,the test dataset-r_squared=0.793252,the corresponding degree=8
    Coefficient:  [ 0.00000000e+00  9.86819406e+01  3.00767055e+00  5.40848852e+00
     -1.58746106e+01  1.91477169e+00  1.03404514e-02  8.86030544e-02
      9.36082835e-03]
    __________________________________________________
    The training dataset-r_squared=0.553865,the test dataset-r_squared=0.791512,the corresponding degree=9
    Coefficient: [ 0.00000000e+00  9.78869938e+01  3.81158047e+00  6.77240719e+00
     -1.59105045e+01  1.64244779e+00 -1.60745564e-02  7.80528099e-02
      8.09780189e-03  2.79641779e-03]
    __________________________________________________
    The training dataset-r_squared=0.554258,the test dataset-r_squared=0.790384,the corresponding degree=10
    Coefficient: [ 0.00000000e+00  9.74913733e+01  4.39934176e+00  7.47532494e+00
     -1.59828872e+01  1.51928877e+00 -3.08243010e-02  7.37043881e-02
      7.49203259e-03  2.66873015e-03  4.10414798e-04]
    __________________________________________________
    The training dataset-r_squared=0.554556,the test dataset-r_squared=0.789517,the corresponding degree=11
    Coefficient:  [ 0.00000000e+00  9.71471289e+01  4.76673465e+00  8.04335499e+00
     -1.60031756e+01  1.41930136e+00 -4.17655820e-02  7.02748057e-02
      7.02488195e-03  2.56631151e-03  3.94086203e-04  9.12164731e-05]
    __________________________________________________
    The training dataset-r_squared=0.554746,the test dataset-r_squared=0.788932,the corresponding degree=12
    Coefficient: [ 0.00000000e+00  9.69412244e+01  5.03208202e+00  8.38856640e+00
     -1.60288096e+01  1.36176783e+00 -4.83881208e-02  6.82821981e-02
      6.74644474e-03  2.50588500e-03  3.84360223e-04  8.93139495e-05
      1.58097622e-05]
    __________________________________________________
    The training dataset-r_squared=0.554878,the test dataset-r_squared=0.788525,the corresponding degree=13
    Coefficient:  [ 0.00000000e+00  9.67902373e+01  5.20020621e+00  8.63430433e+00
     -1.60401576e+01  1.32068198e+00 -5.29313538e-02  6.68699237e-02
      6.55130810e-03  2.46287024e-03  3.77472693e-04  8.79566428e-05
      1.55793847e-05  3.14593156e-06]
    __________________________________________________
    The training dataset-r_squared=0.554965,the test dataset-r_squared=0.788249,the corresponding degree=14
    Coefficient:  [ 0.00000000e+00  9.66938208e+01  5.31829404e+00  8.79257337e+00
     -1.60505615e+01  1.29480894e+00 -5.58620279e-02  6.59872420e-02
      6.42723375e-03  2.43555558e-03  3.73089611e-04  8.70931670e-05
      1.54326409e-05  3.11843117e-06  5.72801426e-07]
    __________________________________________________
    The training dataset-r_squared=0.555024,the test dataset-r_squared=0.788065,the corresponding degree=15
    Coefficient:  [ 0.00000000e+00  9.66281005e+01  5.39448929e+00  8.89912644e+00
     -1.60563919e+01  1.27744308e+00 -5.78371064e-02  6.53875403e-02
      6.34304681e-03  2.41707918e-03  3.70114582e-04  8.65078199e-05
      1.53329695e-05  3.09976035e-06  5.69537369e-07  1.08937732e-07]
    __________________________________________________
    


<a href=""><img src="./imgs/8_8.png" height="auto" width="auto" title="caDesign"></a>


    __________________________________________________
    In the interval [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15],the largest r_squared=0.812928,the corresponding degree=4
    

### 1.4 Gradient Descent
In the regression part, the method of $\widehat{ \beta } = ( X^{'} X)^{-1} X^{'}Y$ matrix calculation is used to solve the model parameter value(namely the regression coefficient). The calculation of the matrix is very complicated, and in some cases, the inverse cannot be obtained. Therefore, another method to estimate the model parameter's optimal value, namely the gradient descent method, is introduced. At present, there is a lot of literature on the interpretation of the gradient method, which mainly includes the illustration by image schema and derivation process of the method, but not understands the details of the actual calculation; for pure formula derivation, there is no example, only by hieroglyphics, can not be implemented. In the online course Udacity, there is an article 'Gradient Descent - Problem of Hiking Down a Mountain' which explains the gradient descent method, including illustrations, derivation, and examples. The explanatory is very clear. Therefore, this article is mainly used to understand the gradient descent method step by step, which is of great help to machine learning and deep learning.

The most common description of the gradient descent method is to go down to the valley, that is, to find the lowest point, which is the minimum value of the loss function. In the process of downhill, every step is also trying to find the steepest path leading to the downhill mountain, that is, to find the gradient at the given point, the direction of the gradient is the direction of the fastest change of the function, and the reverse direction of the gradient can make the function value decline the fastest. And then you keep repeating the process; you keep getting to local minima, and eventually, you get to the valley. Fast to the valley, it is necessary to measure each step in the search for the direction of the steepest measurement of the distance, if too many steps might miss the final valley point, if the pace is too small, increase the calculation duration, and may be trapped in local lows. Hence, the span of each step is proportional to the steepness of the current terrain. If it is steep, take large strides; if it is not, take small steps. The gradient descent method is an optimization algorithm to estimate the local minimum of a function.

#### 1.4.1 Gradients, differentials, and derivatives

The understanding of differentiation can be extended to the slope of a tangent line at a point and the rate of change of the function in a function image. The slope of the tangent is understood as the derivative; the derivative is the slope of the function image at a certain point, namely the ratio of the ordinate increment($\triangle y$) and the abscissa increment ($\triangle x$) at $\triangle x \mapsto 0$. It is generally defined as assuming the defined domain and values of  a function $y=f(x)$ in the real number domain, if $f(x)$ is defined in a neighborhood of the point $x_{0} $, then when independent variable $x$ gets incremental $ \triangle x$ (point $ x_{0} + \triangle x$ is still in the neighborhood)at $x_{0} $, the corresponding $y$ gets the incremental $\triangle y=f( x_{0}+ \triangle x )-f( x_{0} )$, If the limit of the ratio of $\triangle y$ and $\triangle x$ exists at $\triangle x \mapsto 0$, the function $y=f(x)$ is differential at the point $x_{0} $, and the limit is the derivative of the function $y=f(x)$ at the point $x_{0} $, denoted as $f' ( x_{0} )$, that is $f' ( x_{0} )= \lim_{ \triangle x \rightarrow 0} \frac{ \triangle y}{ \triangle x} =\lim_{ \triangle x \rightarrow 0} \frac{f( x_{0}+ \triangle x )-f( x_{0} )}{ \triangle x} $, also available as $y'( x_{0} ), { \frac{dy}{dx} |} _{x= x_{0} } , \frac{dy}{dx} ( x_{0} ),{ \frac{df}{dx} |} _{x= x_{0} }$, etc.

If understood as the change rate of a function, then it is differential, which has been explained in the differential part; differential is a linear description of a function's local change rate. It can approximately describe how the function's value changes when the value of the function independent variable is small enough; it is defined as supposing the function $y=f(x)$ is defined in a certain neighborhood, for a point $x_{0} $ in the neighborhood, when $x_{0} $ changes to the nearby $ x_{0} + \triangle x$(also in the neighborhood), if the incremental $\triangle y=f( x_{0}+ \triangle x )-f( x_{0} )$ can be expressed as $\triangle y=A \triangle x+ o ( \triangle x)$, $A$ is a constant does not depend on $\triangle x$, $o ( \triangle x)$ is an infinitesimal of higher order than $\triangle x$, then call the function $f(x)$ at the point $x_{0} $ differentiable, and $=A \triangle x$ called function at the point $x_{0} $ corresponding to the independent variable increment $\triangle x $ derivative, denoted as $dy$, that is, $dy=A \triangle x$, $dy$ is the linear principal part of $\triangle y$. Usually the increment $\triangle x$ of the independent variable $x$ is called the derivative of the independent variable, called $dx$, that is $dx=\triangle x$.

Differential and derivative are two different concepts. But for unary functions, differentiability and derivable are completely equivalent. The differential of a differentiable function is equal to the derivative times the differential $dx$ of the independent variable, that is, the quotient of the differential of the function with the differential of the independent variable is equal to the derivative of the function, so the derivative is also called the differential quotient. The differential of the function $y=f(x)$ is also denoted as $dy= f' (x)dx$.

The gradient is, in fact, multivariable function derivative( or differential), with $\theta $ as function $J( \theta _{1}, \theta _{2},\theta _{3})$ variable, its $J( \Theta )=0.55-(5 \theta _{1}+ 2 \theta _{2}-12 \theta _{3})$, then $ \triangle J( \Theta )=\langle \frac{ \partial J}{ \partial \theta _{1} }, \frac{ \partial J}{ \partial \theta _{2} }, \frac{ \partial J}{ \partial \theta _{3} } \rangle=\langle -5,-2,12 \rangle$, $ \triangle $ is a symbol of gradient, $\partial$ symbol is used to indicate partial differential(partial meaning), that is, the gradient is to obtain partial differential of each variable respectively, and the gradient is also enclosed in $\langle \rangle$ to indicate that the gradient is a vector.


The gradient is an important concept in calculus; in a function of one variable, the gradient is the differential of the function(or the derivative of the function), representing the tangent line's slope at a given point. In a multivariable function, the gradient is a direction (vector direction). The direction of the gradient indicates the direction in which the function rises fastest at a given point. The opposite direction is the direction in which the function falls fastest at a given point.

> The derivative is the derivative of a function with an independent variable; A partial derivative is the derivative of a function with more than one independent variable. A partial differential is similar to a partial derivative, in that it differentiates one independent variable of a function containing more than one independent variable.

#### 1.4.2 Mathematical interpretation of gradient descent algorithm
The formula is: $ \Theta ^{1} = \Theta ^{0}- \alpha \nabla J( \Theta )$，evaluated at $\Theta ^{0}$, $J$ is the function of $\Theta $, $ \Theta ^{0}$ is the current position, go from this point to the minimum value point  $\Theta ^{1}$ of $J(\Theta)$, the direction is the reverse of the gradient, and use $\alpha$ hyper-parameter(learning rate or step size) to control each step distance to avoid missing the lowest point due to a large step. As it approaches the lowest point, the control of the learning rate becomes more important. The gradient is the direction of the fastest rise; if you go down, the $-$ minus needs to be added in front of the gradient.

* Gradient descent of a function of one variable

The function is defined as: $J(x)= x^{2} $, differentiates the funtion $x$, the result is: $J' =2x$, so the gradient descent formula is: $x_{next}= x_{current}- \alpha *2x$, at the same time, the interation is given. It is convenient to check the magnitude of each iteration by printing the chart.


```python
import sympy
import matplotlib.pyplot as plt
import numpy as np
# Define a function of one variable and draw a curve
x=sympy.symbols('x')
J=1*x**2
J_=sympy.lambdify(x,J,"numpy")

fig, axs=plt.subplots(1,2,figsize=(25,12))
x_val=np.arange(-1.2,1.3,0.1)
axs[0].plot(x_val,J_(x_val),label='J function')
axs[1].plot(x_val,J_(x_val),label='J function')

#Differentiate the function
dy=sympy.diff(J)
dy_=sympy.lambdify(x,dy,"math")

#Initialize
x_0=1 #Initialize the start point
a=0.1 #Configure the learning rate
iteration=15 #Initialize the number of iteration

axs[0].scatter(x_0,J_(x_0),label='starting point')
axs[1].scatter(x_0,J_(x_0),label='starting point')

#Iterate calculation according to the gradient descent formula
for i in range(iteration):
    if i==0:
        x_next=x_0-a*dy_(x_0)
    x_next=x_next-a*dy_(x_next)    
    axs[0].scatter(x_next,J_(x_next),label='epoch=%d'%i)

#Adjust the learning rate and compare the gradient descent rate
a_=0.2
for i in range(iteration):
    if i==0:
        x_next=x_0-a_*dy_(x_0)
    x_next=x_next-a_*dy_(x_next)    
    axs[1].scatter(x_next,J_(x_next),label='epoch=%d'%i)
    
axs[0].set(xlabel='x',ylabel='y')
axs[1].set(xlabel='x',ylabel='y')
axs[0].legend(loc='lower right', frameon=False)  
axs[1].legend(loc='lower right', frameon=False)  
plt.show()  
```


<a href=""><img src="./imgs/8_9.png" height="auto" width="auto" title="caDesign"></a>


The previous part of the code is a function given the regression coefficient, whose coefficient is 1, and the gradient descent algorithm is used to check the trend change of the gradient descent. Note that the $J(x)= x^{2} $ function defined above is the loss function(can be written as $J( \beta )= \beta ^{2} $, $ \beta$ is the regression coefficient), not the model (for example, the regression equation). Differential initially to the loss function, and make the result is 0, to solve the regression coefficient, adjusted for differential at some point in the loss function curve first, find the descending direction and magnitude of this point(vector), and multiplied by the learning rate(adjust the descending rate), and then move on to the next point according to the vector. So,  until a position on which the downward trend is close to 0 is found, the position is required by the model coefficient. According to the above statement, the following code is completed, namely, first define the model; this model is the regression model, to distinguish from the above loss function model, define it as $y= \omega x^{2} $, and assume $\omega=3$, to establish the dataset, the explanatory variable $X$, and the response variable $y$. The model's function is `model_quadraticLinear(w,x)`, expressed and calculated using the Sympy library. Once the model is in place, the loss function can be calculated according to the model and dataset. Here, MSE is used as the loss function, and the calculated result is $MSE=1.75435200000001 (0.333333333333333w-1)^{2} +0.431208 (0.333333333333333w-1)^{2} $. Then, the gradient descent function is defined, namely differentiating MSE $\omega$. After adding the learning rate, the calculated result is $G=0.0728520000000003w - 0.218556000000001$. With these three functions in place, we can begin to train the model, the order is, define the function -->define the loss function -->define the gradient descent --> specify  $\omega=5$ initial value, the loss function is used to calculate the residual sum of squares, namely MSE -->compare whether MSE meets the preset accuracy 'accuracy=1e-5', if not satisfied to start the cycle --> the next point approaching 0 is calculated by the gradient descent formula, and MSE is calculated, and compare whether MSE meets the requirements, run in the cycle --> until 'L<accuracy' meet the requirement, out of the loop, then 'w_next' is the value of model coefficient $\omega$. The calculation result is 'w=3.008625', approximately 3, exactly the assumed value originally used to generate the dataset.



```python
# Define the dataset, and follow the function y= 3*x**2, facilitate to compare the calculated results.
import sympy
from sympy import pprint
import numpy as np
x=sympy.symbols('x')
y=3*x**2
y_=sympy.lambdify(x,y,"numpy")

X=np.arange(-1.2,1.3,0.1)
y=y_(X)
n_size=X.shape[0]

#Initialize
a=0.15 #Configure the learning rate
accuracy=1e-5 #Given the precision
w_=5 #Randomly initialize the coefficient.

#Define the model
def model_quadraticLinear(w,x):
    '''Define a quadratic equation with no intercept b'''    
    return w*x**2

#Define the loss function
def Loss_MSE(model,X,y):
    '''The mean square error(MSE) is used as the loss function'''
    model_=sympy.lambdify(x,model,"numpy")
    loss=(model_(X)-y)**2
    return loss.sum()/n_size/2
    
#To define the gradient descent function is to find the gradient of the loss function.
def gradientDescent(loss,a,w):
    '''The gradient descent function is defined, that is, the differentiation of model variables.'''
    return a*sympy.diff(loss,w)

#Training model
def train(X,y,a,w_,accuracy):
    '''According to the precision value, training model'''
    x,w=sympy.symbols(['x','w'])
    model=model_quadraticLinear(w,x)
    print("Define a function:")
    pprint(model)
    loss=Loss_MSE(model,X,y)
    print("Define the loss function:")
    pprint(loss)
    grad=gradientDescent(loss,a,w)
    print("Define the gradient descent:")
    pprint(grad)
    print("_"*50)
    grad_=sympy.lambdify(w,grad,"math")
    w_next=w_-grad_(w_)
    loss_=sympy.lambdify(w,loss,"math")
    L=loss_(w_next)
    
    i=0
    print("Iterate gradient descent until the result calculated by the loss function is less than the preset value, where w is the weight value(the coefficient of the regression equation)
")
    while not L<accuracy:
        w_next=w_next-grad_(w_next)
        L=loss_(w_next)
        if i%10==0: 
            print("iteration:%d,Loss=%.6f,w=%.6f"%(i,L,w_next))
        i+=1
        #if i%100:break
    return w_next
w_next=train(X,y,a,w_,accuracy)
```

    Define a function:
       2
    w⋅x 
    Define the loss function:
                                              2                                   
    1.75435200000001⋅(0.333333333333333⋅w - 1)  + 0.431208⋅(0.333333333333333⋅w - 
    
      2
    1) 
    Define the gradient descent:
    0.0728520000000003⋅w - 0.218556000000001
    __________________________________________________
    Iterate gradient descent until the result calculated by the loss function is less than the preset value, where w is the weight value(the coefficient of the regression equation)
    iteration:0,Loss=0.717755,w=4.719207
    iteration:10,Loss=0.158109,w=3.806898
    iteration:20,Loss=0.034829,w=3.378712
    iteration:30,Loss=0.007672,w=3.177746
    iteration:40,Loss=0.001690,w=3.083424
    iteration:50,Loss=0.000372,w=3.039154
    iteration:60,Loss=0.000082,w=3.018377
    iteration:70,Loss=0.000018,w=3.008625
    

* Gradient descent of a multivariable function

The gradient descent of a multivariable function is similar to that of a univariable function. The gradient is calculated for each variable separately, and there is no interference between the two. The results are as follows.


```python
import sympy
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
# Define a univariable function, and draw a curve
x1,x2=sympy.symbols(['x1','x2'])
J=x1**2+x2**2
J_=sympy.lambdify([x1,x2],J,"numpy")

x1_val=np.arange(-5,5,0.1)
x2_val=np.arange(-5,5,0.1)
x1_mesh,x2_mesh=np.meshgrid(x1_val,x2_val)
y_mesh=J_(x1_mesh,x2_mesh)

fig, axs=plt.subplots(1,2,figsize=(25,12))
axs[0]=fig.add_subplot(1,2,1, projection='3d')
surf=axs[0].plot_surface(x1_mesh,x2_mesh ,y_mesh, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5,)
axs[1]=fig.add_subplot(1,2,2, projection='3d')
surf=axs[1].plot_surface(x1_mesh,x2_mesh ,y_mesh, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5,)
#fig.colorbar(surf, shrink=0.5, aspect=5)

#Defferentiate the function x1,x2
dx1=sympy.diff(J,x1)
dx2=sympy.diff(J,x2)
dx1_=sympy.lambdify(x1,dx1,"math")
dx2_=sympy.lambdify(x2,dx2,"math")

#Initialize
x1_0=4 #Initialize the x1 starting point
x2_0=4 #Initialize the x2 starting point
iteration=15 #Initialize the number of iteration
a=0.1 #Configure the learning rate

axs[0].scatter(x1_0,x2_0,J_(x1_0,x2_0),label='starting point',c='black',s=80)
axs[1].scatter(x1_0,x2_0,J_(x1_0,x2_0),label='starting point',c='black',s=80)

#Iterate calculation according to the gradient descent
for i in range(iteration):
    if i==0:
        x1_next=x1_0-a*dx1_(x1_0)
        x2_next=x2_0-a*dx2_(x2_0)
        
    x1_next=x1_next-a*dx1_(x1_next)    
    x2_next=x2_next-a*dx2_(x2_next)    
    axs[0].scatter(x1_next,x2_next,J_(x1_next,x2_next),label='epoch=%d'%i,s=80)
    
#Adjust the learning rate and compare the gradient descent rate
a_=0.2
for i in range(iteration):
    if i==0:
        x1_next=x1_0-a_*dx1_(x1_0)
        x2_next=x2_0-a_*dx2_(x2_0)
        
    x1_next=x1_next-a_*dx1_(x1_next)    
    x2_next=x2_next-a_*dx2_(x2_next)    
    axs[1].scatter(x1_next,x2_next,J_(x1_next,x2_next),label='epoch=%d'%i,s=80)

    
axs[0].set(xlabel='x',ylabel='y',zlabel='z')
axs[1].set(xlabel='x',ylabel='y')
axs[0].legend(loc='lower right', frameon=False)  
axs[1].legend(loc='lower right', frameon=False)  

axs[0].view_init(60,200) #The angle of the graph can be rotated for easy observation.
axs[1].view_init(60,200)
plt.show()  
```


<a href=""><img src="./imgs/8_10.png" height="auto" width="auto" title="caDesign"></a>


Solving the binary function model with the gradient descent method is the same as the above on the unary quadratic function model. Note that in the following solution process, the learning rate configuration has a great influence on the calculation results, so we can try different learning rates to observe the change of the regression coefficient. The calculation results under the condition of $\alpha =0.01$ are $w=3.96$，$v=4.04$; there is still a distance away from the hypothesis of 3, 5. We can also print graphs to observe the difference between the real plane and the trained model's plane, and normalize it by adding penalty L2 or L1 to try to improve it, but this already gives you a clear understanding of the gradient descent algorithm, which is a very important basis for the future application of the [Sklear](https://scikit-learn.org/stable/) machine learning library and the [Pytorch](https://pytorch.org/) deep learning library.


At the same time, the SGDRegressor( Stochastic Gradient Descent) provided by the Sklearn library is also used to train the data; the results are $w=2.9999586$，$v=4.99993523$, about 3 and 5, which are the same as the coefficient values of the original hypothesis.


```python
# Define the dataset and follow the function y= 3*x**2 to compare the calculation results.
import sympy
from sympy import pprint
import numpy as np
x1,x2=sympy.symbols(['x1','x2'])
y=3*x1+5*x2
y_=sympy.lambdify([x1,x2],y,"numpy")

X1_val=np.arange(-5,5,0.1)
X2_val=np.arange(-5,5,0.1)
y_val=y_(X1_val,X2_val)
n_size=y_val.shape[0]

#Initialize
a=0.01 #Configure the learning rate
accuracy=1e-10 #Given the precision
w_=5 #The random initialization coefficient corresponds to the x1 coefficient.
v_=5 #The random initialization coefficient corresponds to the x2 coefficient.

#Define the model
def model_quadraticLinear(w,v,x1,x2):
    '''定义二元一次方程，不含截距b'''    
    return w*x1+v*x2

#Define the loss function
def Loss_MSE(model,X1,X2,y):
    '''用均方误差（MSE）作为损失函数'''
    model_=sympy.lambdify([x1,x2],model,"numpy")
    loss=(model_(x1=X1,x2=X2)-y)**2
    return loss.sum()/n_size/2

#To define the gradient descent function is to find the gradient of the loss function.
def gradientDescent(loss,a,w,v):
    '''定义梯度下降函数，即对模型变量微分'''
    return a*sympy.diff(loss,w),a*sympy.diff(loss,v)

#Training model
def train(X1_val,X2_val,y_val,a,w_,v_,accuracy):
    '''According to the precision value, training model''''
    w,v=sympy.symbols(['w','v'])
    model=model_quadraticLinear(w,v,x1,x2)
    print("Define a function:")
    pprint(model)
    loss=Loss_MSE(model,X1_val,X2_val,y_val)
    print("Define the loss function:")
    pprint(loss)    
    grad_w,grad_v=gradientDescent(loss,a,w,v)
    print("Define the gradient descent:")
    pprint(grad_w)
    pprint(grad_v)
    print("_"*50)    
    gradw_=sympy.lambdify([v,w],grad_w,"math")
    gradv_=sympy.lambdify([v,w],grad_v,"math")
    
    w_next=w_-gradw_(v_,w_)   
    v_next=v_-gradv_(v_,w_)
    loss_=sympy.lambdify([v,w],loss,"math")
    L=loss_(w=w_next,v=v_next)    
    
    i=0
    print("Iterate gradient descent until the result calculated by the loss function is less than the preset value, where w,v is the weight value(the coefficient of the regression equation")
    while not L<accuracy:        
        w_next=w_next-gradw_(w=w_next,v=v_next)
        v_next=v_next-gradv_(v=v_next,w=w_next)
        L=loss_(w=w_next,v=v_next)        
        if i%10==0: 
            print("iteration:%d,Loss=%.6f,w=%.6f,v=%.6f"%(i,L,w_next,v_next))
        i+=1
    return w_next,v_next
w_next,v_next=train(X1_val,X2_val,y_val,a,w_,v_,accuracy)


fx=w_next*x1+v_next*x2
fx_=sympy.lambdify([x1,x2],fx,"numpy")
fig, ax=plt.subplots(figsize=(10,10))
ax=fig.add_subplot( projection='3d')

x1_mesh,x2_mesh=np.meshgrid(X1_val,X2_val)
y_mesh=y_(x1_mesh,x2_mesh)
y_pre_mesh=fx_(x1_mesh,x2_mesh)

surf=ax.plot_surface(x1_mesh,x2_mesh ,y_mesh, cmap=cm.coolwarm,linewidth=0, antialiased=False,alpha=0.5,)
surf=ax.plot_surface(x1_mesh,x2_mesh ,y_pre_mesh, cmap=cm.ocean,linewidth=0, antialiased=False,alpha=0.2,)
#fig.colorbar(surf, shrink=0.5, aspect=5)

#Training with SGDRegressor(Stochastic Gradient Descent) provided by the Sklearn library
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

X_=np.stack((x1_mesh.flatten(),x2_mesh.flatten())).T
y_=y_mesh.flatten()
X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.33, random_state=42)
SDGreg=SGDRegressor(loss='squared_loss') #Configure the loss function, which is regularized, that is, the penalty item defaults to L2.
SDGreg.fit(X_train,y_train)
print("_"*50)
print("Sklearn SGDRegressor test set r-squared score%s"%SDGreg.score(X_test,y_test))
print("Sklearn SGDRegressor coef_:",SDGreg.coef_)

ax.set(xlabel='x1',ylabel='x2',zlabel='z')
ax.view_init(13,200) #The angle of the graph can be rotated for easy observation.
plt.show() 
```

    Define a function:
    v⋅x₂ + w⋅x₁
    Define the loss function:
                                             2                                    
    137.360000000001⋅(-0.125⋅v - 0.125⋅w + 1)  + 129.359999999998⋅(0.125⋅v + 0.125
    
           2
    ⋅w - 1) 
    Define the gradient descent:
    0.0833499999999994⋅v + 0.0833499999999994⋅w - 0.666799999999996
    0.0833499999999994⋅v + 0.0833499999999994⋅w - 0.666799999999996
    __________________________________________________
    Iterate gradient descent until the result calculated by the loss function is less than the preset value, where w,v is the weight value(the coefficient of the regression equation
    iteration:0,Loss=8.172455,w=4.694389,v=4.705967
    iteration:10,Loss=0.251475,w=4.091926,v=4.153720
    iteration:20,Loss=0.007738,w=3.986244,v=4.056846
    iteration:30,Loss=0.000238,w=3.967706,v=4.039853
    iteration:40,Loss=0.000007,w=3.964454,v=4.036872
    iteration:50,Loss=0.000000,w=3.963883,v=4.036349
    iteration:60,Loss=0.000000,w=3.963783,v=4.036258
    iteration:70,Loss=0.000000,w=3.963766,v=4.036241
    __________________________________________________
    Sklearn SGDRegressor test set r-squared score0.9999999998258148
    Sklearn SGDRegressor coef_: [2.9999586  4.99993523]
    


<a href=""><img src="./imgs/8_11.png" height="auto" width="auto" title="caDesign"></a>


#### 1.4.3 Training public health data with SGDRegressor( Stochastic Gradient Descent) provided by the Sklearn library
Use SGDRegressor to train the multiple regression model. In the parameter setting, the loss function is configured as 'squared_loss', and the penalty item is L2(specific information can be obtained from the official website of Sklearn). The calculation result is that the determination coefficient is 0.76, which is smaller than the polynomial regression. Six coefficients were obtained, corresponding to six features.


```python
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
X_=data_Income[['Per Capita Income','Below Poverty Level','Crowded Housing','Dependency','No High School Diploma','Unemployment',]].to_numpy() #Convert the feature values to the eigenvector matrix of Numpy
y_=data_Income['Childhood Blood Lead Level Screening'].to_numpy() #Convert the target value to a vector in Numpy format

X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.33, random_state=42)
SDGreg=make_pipeline(StandardScaler(),
                     SGDRegressor(loss='squared_loss',max_iter=1000, tol=1e-3))
SDGreg.fit(X_train,y_train)
print("_"*50)
print("Sklearn SGDRegressor test set r-squared score%s"%SDGreg.score(X_test,y_test))
print("Sklearn SGDRegressor coef_:",SDGreg[1].coef_)
```

    __________________________________________________
    Sklearn SGDRegressor test set r-squared score0.7608544825019922
    Sklearn SGDRegressor coef_: [ 3.72006337 35.76012163 27.6430782   5.88146901 42.79059143 15.77942562]
    

> In terms of the loss function, and the cost function

 Loss Function, For a single sample, measure the difference between the predicted value $\widehat{y} ^{(i)} $ and the observed value $y^{(i)} $;
 
 Cost Function, For multiple samples, measure the difference between the predicted value $\sum_{i=1}^n \widehat{y} ^{(i)} $ of multiple samples and the observed value $\sum_{i=1}^n y^{(i)} $.
 
The two's division is not reflected in the actual use of relevant literature, often confused, so there is no special definition.

### 1.5 key point
#### 1.5.1 data processing technique

* Use the Sklearn library to split dataset, standardize, regularize(Ridge, Lasso), make a pipeline.

* Sklearn model, simple linear regression, k-NN, polynomial regression, stochastic gradient descent algorithm

* Sklearn model accuracy evaluation, determination coefficient, mean absolute error, mean square error

* PySAL library 'pointpats' Point Pattern analysis method

#### 1.5.2 The newly created function tool
* function - Returns the nearest point coordinates for a specified number of neighbors,`k_neighbors_entire(xy,k=3)`

* function - Polynomial regression degree selection and regularization, `PolynomialFeatures_regularization(X,y,regularization='linear')`

* Gradient descent method - define the model, define the loss function, define the gradient descent function, define the training model function

#### 1.5.3 The python libraries that are being imported


```python
import math
import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.polynomial import polyutils as pu

import matplotlib.pyplot as plt
from matplotlib.text import OffsetFrom
from matplotlib import cm
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
    
import sympy
from sympy import pprint
from warnings import filterwarnings
```

#### 1.5.4 Reference
1. Cavin Hackeling. Mastering Machine Learning with scikit-learn[M].Packt Publishing Ltd.July 2017.Second published. 
