> Created on Mon Dec 18 11/43/44 2017  @author: Richie Bao-caDesign(cadesign.cn) __+updated on Sun Jul 19 15/12/45 2020 by Richie Bao

## 1. Simple regression, multiple regression
In the chapter of correlation coefficient analysis, the correlation coefficients between pairs are calculated. Still, it does not distinguish the independent variable, predictor variable, explanatory variable, and the dependent variable, outcome variable, or criterion variable in calculating correlation coefficients. Besides, reflecting the nature and strength of the relationship between variables, regression analysis aims to predict dependent variables based on independent variables. The accuracy of the prediction depends on the degree of correlation; the stronger the correlation is, the more accurate the prediction will be. According to the number of independent variables, a regression can be divided into simple regression and multivariate(multiple) regression. Multivariate regression includes two or more independent variables and one dependent variable. So how much do these independent variables as a whole relate to the dependent variables? As an individual, what is the strength of the relationship between independent variables and dependent variables? At the same time, what is the relative strength between independent variables, and whether there is an interaction effect between independent variables? After answering the above questions, the relationship between independent variables and dependent variables is explained.

The content involved in regression is extremely rich, and the types of regression are rich and colorful. In the practical application of regression to predict the dependent variable does not start from the most basic calculation step by step but directly use an integrated python library,  then goes to the famous machine learning library [scikit-learn](https://scikit-learn.org/stable/index.html), the library contains data preprocessing, dimension reduction, classification model, clustering model, and regression model, and model selection can help us deal with much urban spatial data analysis. At the same time, we can also use [statsmodels](https://www.statsmodels.org/stable/index.html) library, etc. Of course,  starting with an integrated model is detrimental to our understanding of regression, or even any statistical knowledge or data analysis algorithms. It tends to be poorly understood. So step by step,  starting with the basics and using python to sort out the context.

> References for this part:
> 1.Shin Takahashi,Iroha Inoue.The Manga Guide to Regression Analysis.No Starch Press; 1st Edition (May 1, 2016) 
> 2. Timothy C.Urdan.Statistics in Plain English. Routledge; 3rd Edition (May 27, 2010)
> 3. Douglas C.Montgomery,Elizabeth A.Peck,G.Geoffrey Vining. Introduction to linear regression analysis). Wiley; 5th Edition (April 9, 2012)

### 1.1 Preliminary knowledge -- inverse function

#### 1.1.1 Inverse function
If the function $f(x)$ has $y=f(x)$, $f$ is a function of y with respect to x;   If there is function $g$ such that $g(y)=g(f(x))=x$, then $g$ is a function of x with respect to the independent variable y, and $g$ is called the inverse function of $f$. If a function has an inverse function, the function becomes invertible. Call it the function $f$ and its inverse function $f^{-1} $. Hypothesis function $f=2*x+1$, then $x=(f/2-1/2)$, replacing $x$ with $f^{-1}$, and $f$ with $x$, the result is the inverse funciton$f^{-1}=x/2-1/2$. Expressed in the following code assumes that the function and its inverse function, using the algebraic calculation library[sympy](https://docs.sympy.org/latest/index.html), a lightweight 'Sympy' grammar way to maintain the form of the python itself, makes clear, simple math expressions and calculation. When you print the chart $f$ and its inverse function$f^{-1} $, you can see that their relative horizontal and vertical coordinates have been transposed. There are many online automatic conversion platforms to find the inverse function of a function; you can search the input formula to get the formula of its inverse function,for example, [Symbolab](https://www.symbolab.com/). If $f(x)= \frac{ x^{2} +x+1}{x} $, its inverse function is $g(x)= \frac{-1+x+ \sqrt{ x^{2}-2x-3 } }{2} $, and $g(x)= \frac{-1+x- \sqrt{ x^{2}-2x-3 } }{2} $.


```python
import sympy
from sympy import init_printing,pprint,sqrt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

init_printing() #Sympy delivers a variety of formula print patterns.

#Example-A
# Define the symbol
x=sympy.Symbol('x')

# Define expression
f=2*x+1 #function fx
g=x/2-1/2 #The inverse of the fx

#The transformation expression is equivalent to the Numpy function to realize numerical calculation.
x_array=np.arange(-5,10)
f_=sympy.lambdify(x,f,"numpy")
g_=sympy.lambdify(x,g,"numpy")

#Solve the function and plot a diagram 
fig, axs=plt.subplots(1,2,figsize=(16,8))
axs[0].plot(x_array,f_(x_array),'+--',color='tab:blue',label='$f=2*x+1$')
axs[0].plot(f_(x_array),g_(f_(x_array)),'o--',color='tab:red',label='$f^{-1}=x/2-1/2$')

axs[0].set_title('$fx=2*x+1$')
axs[0].legend(loc='upper left', frameon=False)
axs[0].hlines(y=0,xmin=-5,xmax=5,lw=2,color='gray')
axs[0].vlines(x=0,ymin=-5,ymax=5,lw=2,color='gray')


#Example-B
f_B=(x**2+x+1)/x
print("Use 'pprint' to print formula:")
pprint(f_B,use_unicode=True) #Use 'pprint' to print formula:
g_B_negative=(-1+x+sqrt(x**2-2*x-3))/2
g_B_positive=(-1+x-sqrt(x**2-2*x-3))/2

f_B_=sympy.lambdify(x,f_B,"numpy")
g_B_positive_=sympy.lambdify(x,g_B_positive,"numpy")
g_B_negative_=sympy.lambdify(x,g_B_negative,"numpy")

x_B_array=np.arange(-10,21)
x_B_array_positive=x_B_array[x_B_array>0]
axs[1].plot(x_B_array_positive,f_B_(x_B_array_positive),'+',color='tab:blue',label='$+:f(x)$')
axs[1].plot(g_B_positive_(f_B_(x_B_array_positive)),f_B_(x_B_array_positive),'o',color='tab:red',label='$+:g(x)$')

x_B_array_negative=x_B_array[x_B_array<0]
axs[1].plot(x_B_array_negative,f_B_(x_B_array_negative),'+--',color='tab:blue',label='$-:f(x)$')
axs[1].plot(g_B_negative_(f_B_(x_B_array_negative)),f_B_(x_B_array_negative),'o--',color='tab:red',label='$-:g(x)$')

axs[1].hlines(y=0,xmin=-5,xmax=5,lw=2,color='gray')
axs[1].vlines(x=0,ymin=-5,ymax=5,lw=2,color='gray')
axs[1].legend(loc='upper left', frameon=False)
axs[1].set_title('$fx=(x**2+x+1)/x$')

plt.show()
print("JupyterLab direct output formula:g_B_negative=")
g_B_negative #Direct output formula with JupyterLab
```

    Use 'pprint' to print formula:
     2        
    x  + x + 1
    ──────────
        x     
    


<a href=""><img src="./imgs/7_1.png" height="auto" width="auto" title="caDesign"></a>


    JupyterLab direct output formula:g_B_negative=
    




$\displaystyle \frac{x}{2} + \frac{\sqrt{x^{2} - 2 x - 3}}{2} - \frac{1}{2}$



#### 1.1.2 Exponential function and Natural logarithm function
An Exponential function is a mathematical function with the form $b^{x} $, $b$ is the base,  $b$ is index(exponent).

The logarithm is the inverse of the power operation, so if $x=b^{y} $, then $y=log_{b}x $, $b$ is the base of the logarithm, $y$ is the logarithm of $x$, $x$ for the base $b$. Typical bases are $e$、10 or 2.

Natural logarithm is the logarithm function base of the mathematical constant $e$, marked as $lnx$ or $log_{e}x $, and its inverse function is the exponential function $e^{x}$.

* Properties of exponential function and logarithm function:

1. $( e^{a} )^{b} = e^{ab} $
2. $\frac{ e^{a} }{ e^{b} } = e^{a-b} $
3. $a=log( e^{a} )$
4. $log(a^{b} )=b \times (loga)$
5. $loga+logb=log(a \times b)$


```python
import math
from sympy import ln,log,Eq
x=sympy.Symbol('x')
f_exp=2**x
f_exp_=sympy.lambdify(x,f_exp,"numpy")

fig, axs=plt.subplots(1,3,figsize=(25,8))
exp_x=np.arange(-10,10,0.5,dtype=float)
axs[0].plot(exp_x,f_exp_(exp_x),label="f(x)=2**x")
axs[0].legend(loc='upper left', frameon=False)

f_exp_e=math.e**x
f_exp_e_=sympy.lambdify(x,f_exp_e,"numpy")
axs[1].plot(exp_x,f_exp_e_(exp_x),label="f(x)=e**x",color='r')
axs[1].legend(loc='upper left', frameon=False)

log_x=np.arange(1,20,dtype=float)
f_ln=ln(x) 
f_ln_=sympy.lambdify(x,f_ln,"numpy")
axs[2].plot(log_x,f_ln_(log_x),label='base=e')

f_log_2=log(x,2)
f_log_2_=sympy.lambdify(x,f_log_2,"numpy")
axs[2].plot(log_x,f_log_2_(log_x),label="base=2")

f_log_10=log(x,10)
f_log_10_=sympy.lambdify(x,f_log_10,"numpy")
axs[2].plot(log_x,f_log_10_(log_x),label="base=10")
axs[2].legend(loc='upper left', frameon=False)

plt.show()
```


<a href=""><img src="./imgs/7_2.png" height="auto" width="auto" title="caDesign"></a>


#### 1.1.3 Differential
The differential is a linear description of the local change rate of a function. It is an approximate description of how a function value changes when the value of its independent variable change is sufficiently small. Firstly, a dataset was established based on Meiyu's age and height data in the book *The Manga Guide to Regression Analysis* to calculate the correlation coefficient between age and height. The results p_value<0.05, the correlation coefficient Pearson's r=0.942 indicated a strong direct correlation between age and height. Since the correlation between them can build the regression equation, three kinds of the regression model are given in the following code(equations): one kind is $f(x)=- \frac{326.6}{x}+173.3 $ given by *The Manga Guide to Regression Analysis*, the other two are used Linear Models, LinearRegression, and Polynomial regression based on LinearRegression directly provided by Scikit-learn(Sklearn) library. As for the grammar rules of Sklearn, you can refer to the guidance given by the Scikit-learn official website. The grammar structure of Sklearn adheres to the characteristics of python, with strong readability and natural code writing. Of the three regression models, polynomial regression is the best fit, followed by the formula given in  *The Manga Guide to Regression Analysis*. The simple and crude linear regression is different from the shape of the curve of approximate logarithm function due to its linearity.

$y=- \frac{326.6}{x}+173.3 $Differentiating for x, that is, finding the average annual increase in height from $x$ year to $x$ year (independent variable in the year) within a very short period, $\frac{(- \frac{326.6}{x+ \triangle }+173.3 )-(- \frac{326.6}{x} +173.3)}{ \triangle } = \frac{326.6}{ x^{2} } $，Differentials are computed directly using the `diff` tools supplied by Sympy library, and the results are recorded: $\frac{dy}{dx}= \frac{df}{dx}  = y'= f' = \frac{326.6}{ x^{2} }  $。

* Common differential formula:

1. $y=x$，differentiating to $x$, $\frac{dy}{dx}=1$
2. $y= x^{2} $，differentiating to $x$,$\frac{dy}{dx}=2x$
3. $y= \frac{1}{x} $，differentiating to $x$,$\frac{dy}{dx}=- x^{-2} $
4. $y= \frac{1}{ x^{2} } $，differentiating to $x$,$\frac{dy}{dx}=- 2x^{-3} $
5. $y= (5x-7)^{2} $，differentiating to $x$,$\frac{dy}{dx}=2(5x-7)\times 5 $
6. $y= (ax+b)^{n} $，differentiating to $x$,$\frac{dy}{dx}=n (ax+b)^{n-1} \times a  $
7. $y= e^{x} $，differentiating to $x$,$\frac{dy}{dx}=e^{x}$
8. $y=logx$，differentiating to $x$,$\frac{dy}{dx}=\frac{1}{x} $
9. $y=log(ax+b)$，differentiating to $x$,$\frac{dy}{dx}=\frac{1}{ax+b}  \times a $ 
10. $y=log(1+ ea^{x+b} )$，differentiating to $x$,$\frac{dy}{dx}=\frac{1}{1+ e^{ax+b} }  \times a e^{ax+b}  $

In the code field, calculations are performed directly by the 'diff' method in Sympy or other libraries' methods.


```python
import pandas as pd
from scipy import stats
emma_statureAge={"age":list(range(4,20)),"stature":[100.1,107.2,114.1,121.7,126.8,130.9,137.5,143.2,149.4,151.6,154.0,154.6,155.0,155.1,155.3,155.7]}
emma_statureAge_df=pd.DataFrame(emma_statureAge)

r_=stats.pearsonr(emma_statureAge_df.age,emma_statureAge_df.stature)
print(
    "pearson's r:",r_[0],"\n",
    "p_value:",r_[1]
     )

#Raw data scatter diagram.
fig, axs=plt.subplots(1,3,figsize=(25,8))
axs[0].plot(emma_statureAge_df.age,emma_statureAge_df.stature,'o',label='ground truth',color='r')

#A - Use sklearn.linear_model.LinearRegression()，Ordinary least squares Linear Regression in Sklearn-Obtain the regression equation by ordinary least squares linear regression
from sklearn.linear_model import LinearRegression
X=emma_statureAge_df.age.to_numpy().reshape(-1,1)
y=emma_statureAge_df.stature.to_numpy()

#Fitting model
LR=LinearRegression().fit(X,y)

#Model parameters
print("slop:%.2f,intercept:%.2f"%(LR.coef_, LR.intercept_))
print(LR.get_params())

#Model prediction
axs[0].plot(emma_statureAge_df.age,LR.predict(X),'o-',label='linear regression')

#B -  Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
model=Pipeline([('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression(fit_intercept=False))])
reg=model.fit(X,y)
axs[0].plot(emma_statureAge_df.age,reg.predict(X),'+-',label='polynomial regression')

#C - Using the formula given by 'The Manga Guide to Regression Analysis'
from sympy import Symbol
x=Symbol('x')
f_emma=-326.6/x+173.3
f_emma_=sympy.lambdify(x,f_emma,"numpy")
axs[0].plot(emma_statureAge_df.age,f_emma_(emma_statureAge_df.age),'o-',label='$-326.6/x+173.3$')


#
axs[1].plot(emma_statureAge_df.age,emma_statureAge_df.stature,'o',label='ground truth',color='r')
axs[1].plot(emma_statureAge_df.age,f_emma_(emma_statureAge_df.age),'o-',label='$-326.6/x+173.3$')

def demo_con_style(a_coordi,b_coordi,ax,connectionstyle):
    '''
    function - Draw the connection line in the subgraph of the Matplotlib.
    reference - matplotlib official website Connectionstyle Demo
    
    Paras:
    a_coordi - The x, and y coordinate of a
    b_coordi - The x, and y coordinate of b
    ax - subgraph
    connectionstyle - Form of the connection line
    '''
    x1, y1=a_coordi[0],a_coordi[1]
    x2, y2=b_coordi[0],b_coordi[1]

    ax.plot([x1, x2], [y1, y2], ".")
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="0.5",
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle=connectionstyle,
                                ),
                )

    ax.text(.05, .95, connectionstyle.replace(",", ",\n"),
            transform=ax.transAxes, ha="left", va="top")

dx=3
demo_con_style((6,f_emma.evalf(subs={x:6})),(6+dx,f_emma.evalf(subs={x:6+dx})),axs[1],"angle,angleA=-90,angleB=180,rad=0")    
axs[1].text(7, f_emma.evalf(subs={x:6})-3, "△ x", family="monospace",size=20)
axs[1].text(9.3, f_emma.evalf(subs={x:9.3})-10, "△ y", family="monospace",size=20)

#The 'diff' method supplied by Sympy was used for differentiation.
from sympy import diff
print("f_emma=-326.6/x+173.3 differentiating to x:")
pprint(diff(f_emma),use_unicode=True) 
diff_f_emma_=sympy.lambdify(x,diff(f_emma),"numpy")
axs[2].plot(emma_statureAge_df.age,diff_f_emma_(emma_statureAge_df.age),'+--',label='annual growth',color='r')

axs[2].legend(loc='upper right', frameon=False)
axs[1].legend(loc='lower right', frameon=False)
axs[0].legend(loc='upper left', frameon=False)
plt.show()
```

    pearson's r: 0.9422225583501309 
     p_value: 4.943118398567093e-08
    slop:3.78,intercept:94.82
    {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}
    f_emma=-326.6/x+173.3 differentiating to x:
    326.6
    ─────
       2 
      x  
    


<a href=""><img src="./imgs/7_3.png" height="auto" width="auto" title="caDesign"></a>


#### 1.1.4 Matrix
A $m \times n$ matrix is a rectangular array of $m$ rows and $n$ columns elements. The elements of a matrix can be numbers, symbols, or mathematical expressions. Such as:$\begin{bmatrix}1 & 9&-13 \\20 & 5 &-6\end{bmatrix} $，if $\begin{cases} x_{1}+2 x_{2}=-1  \\3 x_{1}+ 4x_{2}=5  \end{cases} $can be written: $\begin{bmatrix}1 & 2 \\3 & 4 \end{bmatrix}  \begin{bmatrix} x_{1}  \\ x_{2} \end{bmatrix} = \begin{bmatrix}-1 \\5 \end{bmatrix} $，ANd if $\begin{cases} x_{1}+2 x_{2} \\3 x_{1}+ 4x_{2} \end{cases}$，we can write:$\begin{bmatrix}1 & 2 \\3 & 4 \end{bmatrix}  \begin{bmatrix} x_{1}  \\ x_{2} \end{bmatrix} $ 。

Matrix operations may be performed directly using the Matrices section of the Sympy library or other libraries. For more information, refer to the official tutorial, or the corresponding section, which will not be covered here.


```python
from sympy import Matrix,init_printing,pprint
init_printing()
M_a=Matrix([[1, -1], [3, 4], [0, 2]])
pprint(M_a)
```

    ⎡1  -1⎤
    ⎢     ⎥
    ⎢3  4 ⎥
    ⎢     ⎥
    ⎣0  2 ⎦
    

### 1.2 Simple linear regression
In statistics, linear regression is a kind of regression analysis. The relationship between one or more independent variables and dependent variables is modeled using a linear regression equation's least-squares function.  Such a function is a linear combination of one or more model parameters called regression coefficients. A condition with only one independent variable is known as simple linear regression, while a condition with a more independent variable is known as multivariable linear regression.

* The flow of regression analysis:

1. To discuss the significance of solving regression equations, scatter plots of independent and dependent variables are drawn;
2. Solve the regression equation;
3. Confirm the accuracy of the regression equation;
4. To test the regression coefficient;
5. Estimation of population regression$Ax+b$;
6. To make predictions.

#### 1.2.1 Set up dataset
Using the highest temperature ($^{\circ}C$) and iced black tea sales(cups) data from *The Manga Guide to Regression Analysis*, a DataFrame format dataset was first established, indexed by timestamp.



```python
import pandas as pd
import util
from scipy import stats

dt=pd.date_range('2020-07-22', periods=14, freq='D')
dt_temperature_iceTeaSales={"dt":dt,"temperature":[29,28,34,31,25,29,32,31,24,33,25,31,26,30],"iceTeaSales":[77,62,93,84,59,64,80,75,58,91,51,73,65,84]}
iceTea_df=pd.DataFrame(dt_temperature_iceTeaSales).set_index("dt")
util.print_html(iceTea_df,14)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temperature</th>
      <th>iceTeaSales</th>
    </tr>
    <tr>
      <th>dt</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-22</th>
      <td>29</td>
      <td>77</td>
    </tr>
    <tr>
      <th>2020-07-23</th>
      <td>28</td>
      <td>62</td>
    </tr>
    <tr>
      <th>2020-07-24</th>
      <td>34</td>
      <td>93</td>
    </tr>
    <tr>
      <th>2020-07-25</th>
      <td>31</td>
      <td>84</td>
    </tr>
    <tr>
      <th>2020-07-26</th>
      <td>25</td>
      <td>59</td>
    </tr>
    <tr>
      <th>2020-07-27</th>
      <td>29</td>
      <td>64</td>
    </tr>
    <tr>
      <th>2020-07-28</th>
      <td>32</td>
      <td>80</td>
    </tr>
    <tr>
      <th>2020-07-29</th>
      <td>31</td>
      <td>75</td>
    </tr>
    <tr>
      <th>2020-07-30</th>
      <td>24</td>
      <td>58</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>33</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2020-08-01</th>
      <td>25</td>
      <td>51</td>
    </tr>
    <tr>
      <th>2020-08-02</th>
      <td>31</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2020-08-03</th>
      <td>26</td>
      <td>65</td>
    </tr>
    <tr>
      <th>2020-08-04</th>
      <td>30</td>
      <td>84</td>
    </tr>
  </tbody>
</table>



#### 1.2.2 Solving the regression equation
Two methods are used to solve the regression equation; one is a step by step calculation; the other uses the 'LinearRegression' model directly supplied by the Sklearn library. The step by step calculation allows for a deeper understanding of the regression model. After familiarity with the basic calculation process, the Sklearn machine learning library model's direct application will also provide a clearer understanding of various parameters' configuration. Firstly, the correlation coefficient between temperature and sales volume is calculated to confirm a correlation between the two, with its p_value=7.661412804450245e-06, which is less than the significance level of 0.05, the result Pearson's r=0.90 can indicate that the correlation between the two is strong.

It is to make the sum of the difference between all the real and predicted values the smallest to solve the regression equation and find a and b, that is, the sum `S_residual` of the square of all variables' residual `s_residual`  is the smallest. Because temperature is linearly dependent on the sales, using a simple equation: $y=ax+b$, $x$ is the independent variable temperature, $y$ is the dependent variable sale, $a$ and $b$ are the regression coefficients (parameters), respectively called slope and intercept. The solution of a and b can be achieved using the least-squares method, which minimizes the squares of errors(sum of squares of residuals) to find the best functional match for the data. The sum of residual squares is:$(−34𝑎−𝑏+93)^{2} +(−33𝑎−𝑏+91)^{2}+(−32𝑎−𝑏+80)^{2}+(−31𝑎−𝑏+73)^{2}+(−31𝑎−𝑏+75)^{2}+(−31𝑎−𝑏+84)^{2}+(−30𝑎−𝑏+84)^{2}+(−29𝑎−𝑏+64)^{2}+(−29𝑎−𝑏+77)^{2}+(−28𝑎−𝑏+62)^{2}+(−26𝑎−𝑏+65)^{2}+(−25𝑎−𝑏+51)^{2}+(−25𝑎−𝑏+59)^{2}+(−24𝑎−𝑏+58)^{2}$， 

Differentiating to $a$ and $b$ respectively, $\frac{df}{da} $ and $\frac{df}{db} $, is the change quantity of the dependent variable when $\triangle a$(the increment of $a$) and $\triangle b$(the increment of $b$) on the horizontal axis tends to infinitesimal,  that is, infinitely close to $a$ and $b$. The dependent variable is the residual sum of squares that is determined by $a$ and $b$. When $a$and $b$take different values, the residual sum of squares varies accordingly. And suppose the residual sum of squares is 0. In that case, it means that the sum of the difference between the sales predicted by the regression equation of all values of the independent variable temperature and the real value is 0. The difference between the sales predicted by the regression model of single temperature value and the real value tends to 0. In the actual manual calculation, the differential of the residual sum of squares to $a$ and $b$ is to sort out the formula. Finally, the formula for solving the regression equation is: $a= \frac{ S_{xy} }{ S_{xx} } $. $S_{xy}$ is variable `SS_xy`, that is,  the deviation product of $x$ and $y$; $S_{xx}$ is variable `SS_x`, that is, the sum of deviation square of $x$.  After finding $a$, you can derive the formula: $b= \overline{y} - \overline{x} a$, to calculate $b$.


In the python language, the use of related libraries avoids this tedious manual derivation. During the step-by-step calculation, reducing the residual sum of squares in the Sympy library is used as follows:$12020⋅a^{2} + 816⋅a⋅b - 60188⋅a + 14⋅b^{2} - 2032⋅b + 75936$, with direct differentiating of $a$ and $b$ respectively, the results are as follows: $ \frac{df}{da} =24040⋅a + 816⋅b - 60188$ and $ \frac{df}{db} =816⋅a + 28⋅b - 2032$, making these formulas 0, use the 'solve' in Sympy for  binary linear equation group to achieve $a$ and $b$ values.

Finally, using the Sklearn library's LinearRegression model to solve the regression model, you only need a few code lines, and the result is the same as above. You can use the parameters returned by Sklearn to establish the regression equation formula, but this is not done in a practical application. Instead, you apply the regression model in the form of variables to directly predict the value.


```python
import math
import sympy
from sympy import diff,Eq,solveset,solve,simplify
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

r_=stats.pearsonr(iceTea_df.temperature,iceTea_df.iceTeaSales)
print("_"*50)
print(
    "pearson's r:",r_[0],"\n",
    "p_value:",r_[1]
     )
print("_"*50)

#Raw data scatter diagram
fig, axs=plt.subplots(1,3,figsize=(25,8))
axs[0].plot(iceTea_df.temperature,iceTea_df.iceTeaSales,'o',label='ground truth',color='r')
axs[0].set(xlabel='temperature',ylabel='ice tea sales')


#A - Use the 'least square method' to calculate step by step.
#1 - Find the deviation of x and y and the sum of the squares of the deviation
iceTea_df["x_deviation"]=iceTea_df.temperature.apply(lambda row: row-iceTea_df.temperature.mean())
iceTea_df["y_deviation"]=iceTea_df.iceTeaSales.apply(lambda row: row-iceTea_df.iceTeaSales.mean())
iceTea_df["S_x_deviation"]=iceTea_df.temperature.apply(lambda row: math.pow(row-iceTea_df.temperature.mean(),2))
iceTea_df["S_y_deviation"]=iceTea_df.iceTeaSales.apply(lambda row: math.pow(row-iceTea_df.iceTeaSales.mean(),2))
SS_x=iceTea_df["S_x_deviation"].sum()
SS_y=iceTea_df["S_y_deviation"].sum()

#2 - Find the deviation product of x and y and the sum thereof.
iceTea_df["S_xy_deviation"]=iceTea_df.apply(lambda row: (row["temperature"]-iceTea_df.temperature.mean())*(row["iceTeaSales"]-iceTea_df.iceTeaSales.mean()),axis=1)
SS_xy=iceTea_df["S_xy_deviation"].sum()

#3 - The operation process
a,b=sympy.symbols('a b')
iceTea_df["prediciton"]=iceTea_df.temperature.apply(lambda row: a*row+b)
iceTea_df["residual"]=iceTea_df.apply(lambda row: row.iceTeaSales-(a*row.temperature+b),axis=1)
iceTea_df["s_residual"]=iceTea_df.apply(lambda row: (row.iceTeaSales-(a*row.temperature+b))**2,axis=1)
S_residual=iceTea_df["s_residual"].sum()
S_residual_simplify=simplify(S_residual)
print("S_residual simplification(Binary quadratic equation):")
pprint(S_residual_simplify) #The residual sum of squares is a binary quadratic function.
print("_"*50)

#Print the  residual sum of squares graph
S_residual_simplif_=sympy.lambdify([a,b],S_residual_simplify,"numpy")
a_=np.arange(-100,100,5)
a_3d=np.repeat(a_[:,np.newaxis],a_.shape[0],axis=1).T
b_=np.arange(-100,100,5)
b_3d=np.repeat(b_[:,np.newaxis],b_.shape[0],axis=1)
z=S_residual_simplif_(a_3d,b_3d)
from sklearn import preprocessing
z_scaled=preprocessing.scale(z) #Standardize the z value, same as 'from scipy.stats import zscore' method

axs[1]=fig.add_subplot(1,3,2, projection='3d')
axs[1].plot_wireframe(a_3d,b_3d,z_scaled)
axs[1].contour(a_3d,b_3d,z_scaled, zdir='z', offset=-2, cmap=cm.coolwarm)
axs[1].contour(a_3d,b_3d,z_scaled, zdir='x', offset=-100, cmap=cm.coolwarm)
axs[1].contour(a_3d,b_3d,z_scaled, zdir='y', offset=100, cmap=cm.coolwarm)

#4 - Differentiate the residual sum of squares 'S_residual' to a and b, and make it equal to 0.
diff_S_residual_a=diff(S_residual,a)
diff_S_residual_b=diff(S_residual,b)
print("diff_S_residual_a=",)
pprint(diff_S_residual_a)
print("\n")
print("diff_S_residual_b=",)
pprint(diff_S_residual_b)

Eq_residual_a=Eq(diff_S_residual_a,0) #Let the differentiation of a be equal to 0.
Eq_residual_b=Eq(diff_S_residual_b,0) #Let the differentiation of b be equal to 0.
slop_intercept=solve((Eq_residual_a,Eq_residual_b),(a,b)) #Calculate the binary linear equation group
print("_"*50)
print("slop and intercept:\n")
pprint(slop_intercept)
slop=slop_intercept[a]
intercept=slop_intercept[b]

#Slope and intercept are calculated directly using the formula for solving the regression coefficient of the regression equation.
print("_"*50)
slop_=SS_xy/SS_x
print("derivation formula to calculate the slop=",slop_)
intercept_=iceTea_df.iceTeaSales.mean()-iceTea_df.temperature.mean()*slop_
print("derivation formula to calculate the intercept=",intercept_)
print("_"*50)

#5 - A simple linear regression equation is established.
x=sympy.Symbol('x')
fx=slop*x+intercept
print("linear regression_fx=:\n")
pprint(fx)
fx_=sympy.lambdify(x,fx,"numpy")

#Mark the position of a and b on the residual sum of squares.
axs[1].text(slop,intercept,-1.7,"a/b",color="red",size=20)
axs[1].scatter(slop,intercept,-2,color="red",s=80)
axs[1].view_init(60,340) #The angle of the graph can be rotated for easy observation.

#6 - Draw a simple linear regression equation.
axs[0].plot(iceTea_df.temperature,fx_(iceTea_df.temperature),'o-',label='prediction',color='blue')

#Draws lines between the true values and the predicted values
i=0
for t in iceTea_df.temperature:
    axs[0].arrow(t, iceTea_df.iceTeaSales[i], t-t, fx_(t)-iceTea_df.iceTeaSales[i], head_width=0.1, head_length=0.1,color="gray",linestyle="--" )
    i+=1

#B - Use sklearn library, sklearn.linear_model.LinearRegression()，Ordinary least squares Linear Regression-Obtain the regression equation by ordinary least squares linear regression
from sklearn.linear_model import LinearRegression
X,y=iceTea_df.temperature.to_numpy().reshape(-1,1),iceTea_df.iceTeaSales.to_numpy()

#Fitting model
LR=LinearRegression().fit(X,y)
#Model parameters
print("_"*50)
print("Sklearn slop:%.2f,intercept:%.2f"%(LR.coef_, LR.intercept_))
#Model prediction
axs[2].plot(iceTea_df.temperature,iceTea_df.iceTeaSales,'o',label='ground truth',color='r')
axs[2].plot(X,LR.predict(X),'o-',label='linear regression prediction')
axs[2].set(xlabel='temperature',ylabel='ice tea sales')

axs[0].legend(loc='upper left', frameon=False)
axs[2].legend(loc='upper left', frameon=False)

axs[0].set_title('step by step manual calculation')
axs[1].set_title('sum of squares of residuals')
axs[2].set_title('using the Sklearn libray')
plt.show()
util.print_html(iceTea_df,14)
```

    __________________________________________________
    pearson's r: 0.9069229780508894 
     p_value: 7.661412804450245e-06
    __________________________________________________
    S_residual simplification(Binary quadratic equation):
           2                           2                 
    12020⋅a  + 816⋅a⋅b - 60188⋅a + 14⋅b  - 2032⋅b + 75936
    __________________________________________________
    diff_S_residual_a=
    24040⋅a + 816⋅b - 60188
    
    
    diff_S_residual_b=
    816⋅a + 28⋅b - 2032
    __________________________________________________
    slop and intercept:
    
    ⎧   1697     -8254 ⎫
    ⎨a: ────, b: ──────⎬
    ⎩   454       227  ⎭
    __________________________________________________
    derivation formula to calculate the slop= 3.7378854625550666
    derivation formula to calculate the intercept= -36.361233480176224
    __________________________________________________
    linear regression_fx=:
    
    1697⋅x   8254
    ────── - ────
     454     227 
    __________________________________________________
    Sklearn slop:3.74,intercept:-36.36
    


<a href=""><img src="./imgs/7_4.png" height="auto" width="auto" title="caDesign"></a>





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>temperature</th>
      <th>iceTeaSales</th>
      <th>x_deviation</th>
      <th>y_deviation</th>
      <th>S_x_deviation</th>
      <th>S_y_deviation</th>
      <th>S_xy_deviation</th>
      <th>prediciton</th>
      <th>residual</th>
      <th>s_residual</th>
    </tr>
    <tr>
      <th>dt</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-22</th>
      <td>29</td>
      <td>77</td>
      <td>-0.142857</td>
      <td>4.428571</td>
      <td>0.020408</td>
      <td>19.612245</td>
      <td>-0.632653</td>
      <td>29*a + b</td>
      <td>-29*a - b + 77</td>
      <td>(-29*a - b + 77)**2</td>
    </tr>
    <tr>
      <th>2020-07-23</th>
      <td>28</td>
      <td>62</td>
      <td>-1.142857</td>
      <td>-10.571429</td>
      <td>1.306122</td>
      <td>111.755102</td>
      <td>12.081633</td>
      <td>28*a + b</td>
      <td>-28*a - b + 62</td>
      <td>(-28*a - b + 62)**2</td>
    </tr>
    <tr>
      <th>2020-07-24</th>
      <td>34</td>
      <td>93</td>
      <td>4.857143</td>
      <td>20.428571</td>
      <td>23.591837</td>
      <td>417.326531</td>
      <td>99.224490</td>
      <td>34*a + b</td>
      <td>-34*a - b + 93</td>
      <td>(-34*a - b + 93)**2</td>
    </tr>
    <tr>
      <th>2020-07-25</th>
      <td>31</td>
      <td>84</td>
      <td>1.857143</td>
      <td>11.428571</td>
      <td>3.448980</td>
      <td>130.612245</td>
      <td>21.224490</td>
      <td>31*a + b</td>
      <td>-31*a - b + 84</td>
      <td>(-31*a - b + 84)**2</td>
    </tr>
    <tr>
      <th>2020-07-26</th>
      <td>25</td>
      <td>59</td>
      <td>-4.142857</td>
      <td>-13.571429</td>
      <td>17.163265</td>
      <td>184.183673</td>
      <td>56.224490</td>
      <td>25*a + b</td>
      <td>-25*a - b + 59</td>
      <td>(-25*a - b + 59)**2</td>
    </tr>
    <tr>
      <th>2020-07-27</th>
      <td>29</td>
      <td>64</td>
      <td>-0.142857</td>
      <td>-8.571429</td>
      <td>0.020408</td>
      <td>73.469388</td>
      <td>1.224490</td>
      <td>29*a + b</td>
      <td>-29*a - b + 64</td>
      <td>(-29*a - b + 64)**2</td>
    </tr>
    <tr>
      <th>2020-07-28</th>
      <td>32</td>
      <td>80</td>
      <td>2.857143</td>
      <td>7.428571</td>
      <td>8.163265</td>
      <td>55.183673</td>
      <td>21.224490</td>
      <td>32*a + b</td>
      <td>-32*a - b + 80</td>
      <td>(-32*a - b + 80)**2</td>
    </tr>
    <tr>
      <th>2020-07-29</th>
      <td>31</td>
      <td>75</td>
      <td>1.857143</td>
      <td>2.428571</td>
      <td>3.448980</td>
      <td>5.897959</td>
      <td>4.510204</td>
      <td>31*a + b</td>
      <td>-31*a - b + 75</td>
      <td>(-31*a - b + 75)**2</td>
    </tr>
    <tr>
      <th>2020-07-30</th>
      <td>24</td>
      <td>58</td>
      <td>-5.142857</td>
      <td>-14.571429</td>
      <td>26.448980</td>
      <td>212.326531</td>
      <td>74.938776</td>
      <td>24*a + b</td>
      <td>-24*a - b + 58</td>
      <td>(-24*a - b + 58)**2</td>
    </tr>
    <tr>
      <th>2020-07-31</th>
      <td>33</td>
      <td>91</td>
      <td>3.857143</td>
      <td>18.428571</td>
      <td>14.877551</td>
      <td>339.612245</td>
      <td>71.081633</td>
      <td>33*a + b</td>
      <td>-33*a - b + 91</td>
      <td>(-33*a - b + 91)**2</td>
    </tr>
    <tr>
      <th>2020-08-01</th>
      <td>25</td>
      <td>51</td>
      <td>-4.142857</td>
      <td>-21.571429</td>
      <td>17.163265</td>
      <td>465.326531</td>
      <td>89.367347</td>
      <td>25*a + b</td>
      <td>-25*a - b + 51</td>
      <td>(-25*a - b + 51)**2</td>
    </tr>
    <tr>
      <th>2020-08-02</th>
      <td>31</td>
      <td>73</td>
      <td>1.857143</td>
      <td>0.428571</td>
      <td>3.448980</td>
      <td>0.183673</td>
      <td>0.795918</td>
      <td>31*a + b</td>
      <td>-31*a - b + 73</td>
      <td>(-31*a - b + 73)**2</td>
    </tr>
    <tr>
      <th>2020-08-03</th>
      <td>26</td>
      <td>65</td>
      <td>-3.142857</td>
      <td>-7.571429</td>
      <td>9.877551</td>
      <td>57.326531</td>
      <td>23.795918</td>
      <td>26*a + b</td>
      <td>-26*a - b + 65</td>
      <td>(-26*a - b + 65)**2</td>
    </tr>
    <tr>
      <th>2020-08-04</th>
      <td>30</td>
      <td>84</td>
      <td>0.857143</td>
      <td>11.428571</td>
      <td>0.734694</td>
      <td>130.612245</td>
      <td>9.795918</td>
      <td>30*a + b</td>
      <td>-30*a - b + 84</td>
      <td>(-30*a - b + 84)**2</td>
    </tr>
  </tbody>
</table>



#### 1.2.3 Confirm the accuracy of the regression equation
The accuracy of the regression equation(model) is calculated as the coefficient of determination, denoted as $R^{2} $或$r^{2} $, which is used to represent the degree of fitting between the measured value(points in the chart) and the regression equation. Its multiple correlation coefficient is calculated as follows: $R= \frac{\sum_{i=1}^n ( y_{i} - \overline{y} )^{2} ( \widehat{y}_{i} - \overline{ \widehat{y} } )^{2} }{ \sqrt{(\sum_{i=1}^n (y_{i}- \overline{y} )^{2} )(\sum_{i=1}^n ( \widehat{y}_{i} - \overline{ \widehat{y} })^{2} )} } $, $y$ is the observed value, $\overline{y}$ is the mean of the observed value, $\widehat{y}$ is the predicted value, $\overline{ \widehat{y} } $ is the mean of the predicted value. The coefficient of determination $R^{2} $ is the square of the multiple correlation coefficient . The value of the determination coefficient is between 0 and 1. The closer the value is to 1, the higher the accuracy of the regression equation will be. The second calculation formula is: $R^{2} =1- \frac{ SS_{res} }{ SS_{tot} }=1- \frac{ \sum_{i=1}^n e_{i} ^{2} }{SS_{tot}} =1- \frac{ \sum_{i=1}^n (y_{i} - \widehat{y} _{i} )^{2} }{ \sum_{i=1}^n ( y_{i} - \overline{y} )^{2} } $, $SS_{res}$ is the residual sum of squares, $SS_{tot}$ is the sum of the observed values deviation squares(total sum of squares, or total sum of deviation squares), $SS_{tot}$ is residual, $y_{i}$ is the observed value, $\widehat{y}$ is the predicted value, $\overline{y}$ is the mean of observed value. The third is to calcualte directly using the `r2_score` provided by the Sklearn library.

According to the calculation results, the results of the first, second, and third methods are consistent. In subsequent experiments, the calculations are performed directly using the method provided by Sklearn.



```python
def coefficient_of_determination(observed_vals,predicted_vals):
    import pandas as pd
    import numpy as np
    import math
    '''
    function - The determination coefficient of the regression equation
    
    Paras:
    observed_vals - Observed value(measured value)
    predicted_vals - Predicted value
    '''
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    #The sum of the observed value deviation squares(the total sum of squares, or the total sum of deviation squares)
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    #The sum of deviation squares of the predicted value
    pre_mean=vals_df.pre.mean()
    SS_reg=vals_df.pre.apply(lambda row:(row-pre_mean)**2).sum()
    #The sum of the deviation products of the observed value and the predicted value
    SS_obs_pre=vals_df.apply(lambda row:(row.obs-obs_mean)*(row.pre-pre_mean), axis=1).sum()
    
    #The residual sum of squares
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
    
    #The coefficient of determination
    R_square_a=(SS_obs_pre/math.sqrt(SS_tot*SS_reg))**2
    R_square_b=1-SS_res/SS_tot
            
    return R_square_a,R_square_b
    
R_square_a,R_square_b=coefficient_of_determination(iceTea_df.iceTeaSales.to_list(),fx_(iceTea_df.temperature).to_list())   
print("R_square_a=%.5f,R_square_b=%.5f"%(R_square_a,R_square_b))

from sklearn.metrics import r2_score
R_square_=r2_score(iceTea_df.iceTeaSales.to_list(),fx_(iceTea_df.temperature).to_list())
print("using sklearn libray to calculate r2_score=",R_square_)
```

    R_square_a=0.82251,R_square_b=0.82251
    using sklearn libray to calculate r2_score= 0.8225092881166944
    

#### 1.2.4 Test of the regression coefficients(regression significance test) | F distribution and the analysis of variance

In the previous chapters, normal distribution and t-distribution are respectively described. At the same time, F-distribution is a continuous probability distribution widely used in the likelihood ratio test, especially in the analysis of variance(ANOVA), 'scipy.stats.f' is the official case interpretation of F-distribution. The function method is the same as the normal distribution and the t-distribution.


```python
from scipy.stats import f
import matplotlib.pyplot as plt
fig, ax=plt.subplots(1, 1)

dfn, dfd=29, 18
mean, var, skew, kurt=f.stats(dfn, dfd, moments='mvsk')
print("mean=%f, var=%f, skew=%f, kurt=%f"%(mean, var, skew, kurt))

# Plot probability density function,pdf
x=np.linspace(f.ppf(0.01, dfn, dfd),f.ppf(0.99, dfn, dfd), 100) #Take 100 values subject to the degree of freedom dfn and dfd, between 1% and 99%. 
ax.plot(x, f.pdf(x, dfn, dfd),'-', lw=5, alpha=0.6, label='f pdf')

# Fixed distribution shape, namely fixed degrees of freedom.
rv = f(dfn, dfd)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')

vals = f.ppf([0.001, 0.5, 0.999], dfn, dfd)
print("Verify whether the cumulative distribution function (CDF) return value is equal to or approximate to the PPF return value:",np.allclose([0.001, 0.5, 0.999], f.cdf(vals, dfn, dfd)))

#Generate random numbers subject to F-distribution, and plot the histogram.
r=f.rvs(dfn, dfd, size=1000)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()
```

    mean=1.125000, var=0.280557, skew=1.806568, kurt=7.074636
    Verify whether the cumulative distribution function (CDF) return value is equal to or approximate to the PPF return value: True
    


<a href=""><img src="./imgs/7_5.png" height="auto" width="auto" title="caDesign"></a>


* The total sum of squares=regression sum of squares+residual sum of squares

Formula is: $SS_{tot}=\sum_{i=1}^n  ( y_{i} - \overline{y} )^{2}=SS_{reg}+SS_{res}= \sum_{i=1}^n  (\widehat{y} _{i} -    \overline{y} )^{2} + \sum_{i=1}^n  (y_{i} -   \widehat{y} _{i} )^{2} $，In the formula, $SS_{reg}$ is the regression sum of squares;  others are the same as above. The regression sum of squares is the sum of the squares of the difference between the mean of the predicted value(the regression value) and the mean of the observed value(the true value, or the measured value). This statistic reflects the fluctuation of $y$($y_{k} (k=1,2, \ldots ,n)$) caused by the change of the independent variable $x_{1}, x_{2}, \ldots ,x_{m}, $, whose freedom is $df_{reg}=m $, $m$ is the number of independent variable. The unary linear equation of temperature and sales has only one independent variable, so its degree of freedom is 1, that is, only one factor can change freely. The residual sum of squares is the sum of the squares of the difference between the observed value and the predicted value. The existence of residual is caused by experimental error and other factors. The degree of freedom is $df_{res}=n-m-1 $, $n$ is the number of samples, the value of the corresponding $y$. The total deviation sum of squares $y$ has a degree of freedom of $n-1$.

The observed values(samples) are usually given, so the total sum of squared deviations is fixed. The factors that the total sum of squared deviations are the regression sum of squares and the residual sum of squares, respectively, representing the regression equation obtained, or the variation of the value $y$ caused by experimental error and other factors. When the residual sum of squares is smaller(that is, the experimental and other factors have less influence), the greater the squares' sum, the more accurate the regression equation's predicted value is.


> Rediscussion of degrees of freedom (see Wikipedia)
In statistics, the degree of freedom( df) refers to the number of independent or freely changing data in the sample when estimating the population's parameters with the sample's statistics, which is called the degree of freedom of the statistics. Example:
> 1. If there are two dependent variables, $x$ and $y$, if $y=x+c$, $c$ is constant, then its degree of freedom is 1. Because only $x$ can truly change freely, $y$ will be limited by the difference value of $x$;
2. When the average of the population $\mu$ is estimated, since the $n$ values in the sample are independent of each other, any unextracted value is not affected by any extracted value, so the degree of freedom is $n$;
3. The statistic used to estimate the variance of the population $ \sigma ^{2} $ is the variance of the sample $s^{2} $, and $s^{2} $ must be calculated using the average of the sample $\overline{x} $, $\overline{x} $ is determined after the sampling is completed, so as long as $n-1$ is determined in the sample size of $n$, there is only one value of the $n$ that can make the sample conform to $\overline{x} $. So, in other words, only the number of $n-1$ elements in the sample is free to vary, and once the number of $n-1$ is determined, the variance will be determined. Here, the average $\overline{x} $ is equivalent to a constraint. Because of this constraint, the freedom of sample variance $ s^{2} $ is $n-1$;
4. The degree of freedom of the statistical model is equal to the number of independent variables that can be freely evaluated.  For example, suppose a total of $p$ parameters need to be estimated in the regression equation. In that case, $p-1$ independent variables are included(the independent variables corresponding to the intercept are constant), so this regression's freedom is $p-1$.

> In statistics, the unbiased estimator: the standard deviation of a population is usually estimated by a random sample of the population, and its definition is as follows: $s= \sqrt{ \frac{ \sum_{i=1}^n ( x_{i }- \overline{x} ) ^{2} }{n-1} } $, $x_{1}, x_{2} , \ldots , x_{n} $ are samples, $n$ is the sample size, $\overline{x}$ is the mean of the sample. Using $n-1$ instead of $n$, which is called Bessel's correction, corrects the bias in the population variance estimation(which is an estimation using a random sample and is not equal to the population variance), as well as , but not all, of the bias in the population standard deviation estimation. Because the bias depends on a particular distribution, it is not possible to find estimates of the standard deviation for all population distribution that are unbiased. 


* Analysis of variance, ANOVA

The above `the total sum of squares=regression sum of squares+residual sum of squares` is exploring the relationship between the dependent variable(total sum of squares, namely the total sum of squares deviation) and the two factors (or two categories) that affect the change of the dependent variable, namely exploring the relationship between the regression sum of squares and the residual sum of squares, this process is called variance analysis. Before solving the above regression equation, the relationship between temperature and sales volume is not necessarily linear. There may be two situations. One is that the sales volume($y$) fluctuates along a horizontal line regardless of the value of temperature ($x$); Second, temperature and sales has other types of relationship in addition to linear, such as nonlinear.


For the above-obtained regression equation $ f_{x} =ax+b= \frac{1697}{ 454} x- \frac{8254}{227 } $(sample regression model), for the population, $F_{x} =Ax+B$(population regression model), the slop A is about a($A \sim a$), the intercept B is about b($B \sim b$), $\sigma ^{2} = \frac{SS_{res} }{n-2}$(unbiasable estimator, the residual sum of squares has $n-2$ degrees of freedom because the two degrees of freedom are related to the estimated values of the prediced values), The square root of $\sigma ^{2} $ is called the regression standard error sometimes. (The derivation process of $\sigma ^{2} $ obtained from the residual sum of squares can be referred to *Introduction to linear regression analysis*- simple linear regression part).


A very important special case of the population regression equation is $H_{0} :A=0, H_{1} :A \neq 0$, the null hypothesis means that there is no linear relationship between $x$ and $y$, and $x$ is almost useless for explaining the variance of $y$; If the null hypothesis is rejected and the alternative hypothesis is accepted, it means that $x$ is useful for explaining the variance of $y$, it may mean that the linear model is appropriate, but there may also be a nonlinear model that needs to be fitted with a higher-order polynomial. For regression significance tests, either t statistics or a variance analysis can be used. The test of regression coefficients F statistic is: $F_{0} = \frac{ SS_{reg}/ df_{reg} }{ SS_{res}/ df_{res} }$ , $SS_{reg}$ is the regression sum of squares, the degree of freedom $df_{reg} =m$ is 1, $SS_{res}$ is the residual sum of squares, the degree of freedom $df_{res}=n-m-1$ is $14-1-1=12$ (refer to *Introduction to linear regression analysis* for derivation process). If the null hypothesis is true, then the test statistic follows the F-distribution of $m=1$ for the 1st degree of freedom and $n-m-1=12$ for the 2nd degree of freedom. p-value=0.000008, less than the significance level of 0.05, reject the null hypothesis, the alternative hypothesis is true.


```python
def ANOVA(observed_vals,predicted_vals,df_reg,df_res):
    import pandas as pd
    import numpy as np
    import math
    from scipy.stats import f
    '''
    function - Simple linear regression equation - regression significance test(regression coefficient test)
    
    Paras:
    observed_vals - Observed value(measured value)
    predicted_vals - Predicted value
    '''
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    #The deviation sum of squares of the observed values (the total sum of squares, or the total sum of squares of deviations) 
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    
    #The residual sum of squares
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
   
    #The regression sum of squares
    SS_reg=vals_df.pre.apply(lambda row:(row-obs_mean)**2).sum()
    
    print("The total sum of squares=%.6f,The regression sum of squares=%.6f,The residual sum of squares=%.6f"%(SS_tot,SS_reg,SS_res))
    print("The total sum of squares=The regression sum of squares+The residual sum of squares：SS_tot=SS_reg+SS_res=%.6f+%.6f=%.6f"%(SS_reg,SS_res,SS_reg+SS_res))
    
    Fz=(SS_reg/df_reg)/(SS_res/df_res)
    print("F-distribution statistic=%.6f;p-value=%.6f"%(Fz,f.sf(Fz,df_reg,df_res)))

ANOVA(iceTea_df.iceTeaSales.to_list(),fx_(iceTea_df.temperature).to_list(),df_reg=1,df_res=12) 
```

    The total sum of squares=2203.428571,The regression sum of squares=1812.340466,The residual sum of squares=391.088106
    The total sum of squares=The regression sum of squares+The residual sum of squares：SS_tot=SS_reg+SS_res=1812.340466+391.088106=2203.428571
    F-distribution statistic=55.609172;p-value=0.000008
    

The significance test method of the regression equation using the F test is equation analysis. The above process can be reduced to an equation analysis table so that it is easier to trace and clear the vein.

| Statistics       | sum of squares         | degree of freedom  |variance  |variance ratio|
| ------------- |:-------------:| -----:| -----:| -----:|
| regression   | $SS_{reg}= \sum_{i=1}^n  (\widehat{y} _{i} -    \overline{y} )^{2}$ | $df_{reg}=m $|$SS_{reg}/df_{reg}$|  $F_{0} = \frac{ SS_{reg}/ df_{reg}  }{ SS_{res}/ df_{res}  }$    |
| residual      |$SS_{res}= \sum_{i=1}^n  (y_{i} -   \widehat{y} _{i} )^{2}   $   |  $df_{res}= n-m-1$ |$SS_{res}/df_{res}$||
| population| $SS_{tot}=\sum_{i=1}^n  ( y_{i} - \overline{y} )^{2}$     |  $df_{tot}= n-1$  |||


#### 1.2.5 Estimation of the population regression of $Ax+b$---confidence interval estimation
For the regression model of temperature and sales volume, when the temperature is an arbitrary value, the corresponding sales volume is not a fixed value. Still, it follows a normal distribution with an average of $Ax+B$(population regression) and a standard deviation of $\sigma$. Therefore, given the confidence interval (95%，99%, etc.), the population regression of $Ax+B$(that is, the predicted value) is bound to be above a certain value and below a certain value. The confidence interval of the sales corresponding to any temperature is calculated by adding or subtractive an interval from the predicted value. The calculation formula of this interval is: $\sqrt{F(1,n-2;0.05) \times ( \frac{1}{n}+ \frac{ ( x_{i}- \overline{x} )^{2} }{ S_{xx} } ) \times \frac{SS_{res}}{n-2} } $, $n$ is he number of samples, $ x_{i}$ is the sample value of independent variabel(temperature), $\overline{x}$ is the sample mean, $S_{xx}$is the deviation sum of squares of the independence variabel $x$(temperature), $SS_{res}$ is the residual sum of squares.


```python
def confidenceInterval_estimator_LR(x,sample_num,X,y,model,confidence=0.05):
    import numpy as np
    import math
    from scipy.stats import f
    import matplotlib.pyplot as plt
    '''
    function - The simple linear regression confidence interval estimation, and the predicted interval
    
    Paras:
    x - Independent variable value
    sample_num - Sample size
    X - Sample dataset-independent variables
    y - Sample dataset-dependent variables
    model -The linear regression models obtained using Sklearn
    confidence -  the confidence coefficient
    '''
    X_=X.reshape(-1)
    X_mu=X_.mean()
    s_xx=(X_-X_mu)**2
    S_xx=s_xx.sum()
    ss_res=(y-LR.predict(X))**2
    SS_res=ss_res.sum()
    probability_val=f.ppf(q=1-confidence,dfn=1, dfd=sample_num-2) #dfn=1, dfd=sample_num-2
    CI=[math.sqrt(probability_val*(1/sample_num+(x-X_mu)**2/S_xx)*SS_res/(sample_num-2)) for x in X_]
    y_pre=LR.predict(X)
    
    fig, ax=plt.subplots(figsize=(10,10))
    ax.plot(X_,y,'o',label='observations/ground truth',color='r')
    ax.plot(X_,y_pre,'o-',label='linear regression prediction')
    ax.plot(X_,y_pre-CI,'--',label='y_lower')
    ax.plot(X_,y_pre+CI,'--',label='y_upper')
    ax.fill_between(X_, y_pre-CI, y_pre+CI, alpha=0.2,label='95% confidence interval')    
      
    #The predicted interval for a given value
    x_ci=math.sqrt(probability_val*(1/sample_num+(x-X_mu)**2/S_xx)*SS_res/(sample_num-2))
    x_pre=LR.predict(np.array([x]).reshape(-1,1))[0]
    x_lower=x_pre-x_ci
    x_upper=x_pre+x_ci
    print("x prediction=%.6f;confidence interval=[%.6f,%.6f]"%(x_pre,x_lower,x_upper))
    ax.plot(x,x_pre,'x',label='x_prediction',color='r',markersize=20)
    ax.arrow(x, x_pre, 0, x_upper-x_pre, head_width=0.3, head_length=2,color="gray",linestyle="--" ,length_includes_head=True)
    ax.arrow(x, x_pre, 0, x_lower-x_pre, head_width=0.3, head_length=2,color="gray",linestyle="--" ,length_includes_head=True)
        
    ax.set(xlabel='temperature',ylabel='ice tea sales')
    ax.legend(loc='upper left', frameon=False)    
    plt.show()                  
    return CI

sample_num=14
confidence=0.05
iceTea_df_sort=iceTea_df.sort_values(by=['temperature'])
X,y=iceTea_df_sort.temperature.to_numpy().reshape(-1,1),iceTea_df_sort.iceTeaSales.to_numpy()
CI=confidenceInterval_estimator_LR(27,sample_num,X,y,LR,confidence)    
```

    x prediction=64.561674;confidence interval=[60.496215,68.627133]
    


<a href=""><img src="./imgs/7_6.png" height="auto" width="auto" title="caDesign"></a>


#### 1.2.6 Predicted interval
Given a specific value such as temperature is 31, the predicted value is 79.51; however, the actual value is not set as this value, but floats within the confidence interval $[66.060470,92.965962]$ corresponding to the confidence coefficient of 95%, this interval is called the predicted interval.

### 1.3 Multiple linear regression
A regression model with more than one regression variable is called a multivariable(multiple) regression model. If it is a linear model, the latter can be taken as a multivariable linear regression. On many practical problems processing, especially the large data, it will involve a lot of variables, such as the classic Iris dataset in the Sklearn machine learning library contains the independent four variables of the Sepal Length, Sepal Width, Petal Length, and Petal Width. Its dependent variable is the type of iris. Five parameters are needed if a regression model is to be established according to the independent variables and the dependent variable.


#### 1.3.1 Set up dataset
When the python analyzes this part of the content, the relatively simple dataset is still used. The store dataset set in *The Manga Guide to Regression Analysis* is used. The independent variables include the store area($m^{2} $), the nearest station distance(m), and the dependent variable is the monthly turnover(ten thousand yen).


```python
import pandas as pd
import util
from scipy import stats

store_info={"location":['Ill.','Ky.','Lowa.','Wis.','MIch.','Neb.','Ark.','R.I.','N.H.','N.J.'],"area":[10,8,8,5,7,8,7,9,6,9],"distance_to_nearestStation":[80,0,200,200,300,230,40,0,330,180],"monthly_turnover":[469,366,371,208,246,297,363,436,198,364]}
storeInfo_df=pd.DataFrame(store_info)
util.print_html(storeInfo_df,10)
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>location</th>
      <th>area</th>
      <th>distance_to_nearestStation</th>
      <th>monthly_turnover</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ill.</td>
      <td>10</td>
      <td>80</td>
      <td>469</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ky.</td>
      <td>8</td>
      <td>0</td>
      <td>366</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lowa.</td>
      <td>8</td>
      <td>200</td>
      <td>371</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Wis.</td>
      <td>5</td>
      <td>200</td>
      <td>208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MIch.</td>
      <td>7</td>
      <td>300</td>
      <td>246</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Neb.</td>
      <td>8</td>
      <td>230</td>
      <td>297</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ark.</td>
      <td>7</td>
      <td>40</td>
      <td>363</td>
    </tr>
    <tr>
      <th>7</th>
      <td>R.I.</td>
      <td>9</td>
      <td>0</td>
      <td>436</td>
    </tr>
    <tr>
      <th>8</th>
      <td>N.H.</td>
      <td>6</td>
      <td>330</td>
      <td>198</td>
    </tr>
    <tr>
      <th>9</th>
      <td>N.J.</td>
      <td>9</td>
      <td>180</td>
      <td>364</td>
    </tr>
  </tbody>
</table>



#### 1.3.2 Correlation analysis
Correlation analysis is also needed to judge whether the above dataset has the significance of establishing a multiple linear regression model. Because of the increase in variables involved, the correlation coefficient between pairs and the corresponding P-value needs to be calculated. The `correlationAnalysis_multivarialbe` function was established to facilitate correlation analysis for this kind of data type latter. The correlation coefficient between the independent and dependent variables reflects the extent to which the independent variable can explain the dependent variable, and its correlation coefficient is 0.8924,-0.7751. Both independent variables strongly correlate with the dependent variable, which can explain dependent variables and establish a regression model. Simultaneously, the correlation between independent variables can preliminarily judge whether there is multicollinearity between independent variables, that is, there is an accurate correlation or a high correlation between independent variables, which makes the model estimation distorted or difficult to estimate accurately. According to the calculation results, the correlation coefficient between the two independent variables is -0.4922. Still, the corresponding P-value is 0.1485, which means the null hypothesis is rejected, indicating no linear correlation between the two independent variables. Therefore, using the two independent variables to explain the dependent variables at the same time will not distort the regression model.


```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(storeInfo_df)
plt.show()

def correlationAnalysis_multivarialbe(df):
    from scipy.stats import pearsonr
    import pandas as pd
    '''
    function - DataFrame format data, group calculated the Pearson correlation coefficient
    
    Paras:
    df - DataFrame format dataset
    '''
    df=df.dropna()._get_numeric_data()
    df_cols=pd.DataFrame(columns=df.columns)
    p_values=df_cols.transpose().join(df_cols, how='outer')
    correlation=df_cols.transpose().join(df_cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            p_values[r][c]=round(pearsonr(df[r], df[c])[1], 4)
            correlation[r][c]=round(pearsonr(df[r], df[c])[0], 4)
            
            
    return p_values,correlation
p_values,correlation=correlationAnalysis_multivarialbe(storeInfo_df)

print("p_values:")
print(p_values)
print("_"*78)
print("correlation:")
print(correlation)
```


<a href=""><img src="./imgs/7_7.png" height="auto" width="auto" title="caDesign"></a>


    p_values:
                                  area distance_to_nearestStation monthly_turnover
    area                             0                     0.1485           0.0005
    distance_to_nearestStation  0.1485                          0           0.0084
    monthly_turnover            0.0005                     0.0084                0
    ______________________________________________________________________________
    correlation:
                                  area distance_to_nearestStation monthly_turnover
    area                             1                    -0.4922           0.8924
    distance_to_nearestStation -0.4922                          1          -0.7751
    monthly_turnover            0.8924                    -0.7751                1
    

Because there are three variables involved, you can use the Ternary Plot provided by the Plotly library to see the distribution between the two independent variables and the one dependent variable. The value range of variables may vary greatly. When a ternary graph is plotted, some variables' value may be close to the graph's edge, and the relationship between variables cannot be clearly expressed. So using: $\frac{ x_{i} - \overline{x} }{ x_{max}- x_{min} } $ method to standardize each variables respectively. It can be seen from the diagram that the shop area(the color represents the area) gradually increases, and the monthly turnover gradually increases(the size of the points represent the monthly turnover). And the nearest station distance gradually decreased, the monthly turnover gradually increased.


```python
pd.options.mode.chained_assignment = None

columns=['area','distance_to_nearestStation','monthly_turnover']
storeInfo_plot=storeInfo_df[columns]
normalize_df=storeInfo_plot.T.apply(lambda row:(row-row.min())/(row.max()-row.min()) , axis=1).T
normalize_df["location"]=storeInfo_df.location

import plotly.express as px
fig=px.scatter_ternary(normalize_df,a="monthly_turnover", b="area",c="distance_to_nearestStation",hover_name="location",
                       color="area",size="monthly_turnover", size_max=8) 

fig.show()

```

<a href=""><img src="./imgs/7_8.png" height="auto" width="auto" title="caDesign"></a>

#### 1.3.3 Solve multiple regression equation
The method to solve the multiple regression equation is the same as the simple linear regression method. The least-square method is used to solve the partial regression coefficient. In the solution process, three methods are used. The first is to differentiate $a1$, $a2$ and $b$ in Sympy library for the residual sum of squares $SS_res$ respectively. When the respective differential value is equal to 0, the reflected residual sum of squares is 0, that is, the sum of squares of the difference between the observed value and the predicted value is 0. In contrast, the difference between the single observed value and the corresponding predicted value tends to 0; Secondly, the parameters are solved by matrix calculation. The calculation formula is as follows:$\widehat{ \beta } = ( X^{'} X)^{-1} X^{'}y$，where,$X=\left[\begin{matrix}1 & 10 & 80\\1 & 8 & 0\\1 & 8 & 200\\1 & 5 & 200\\1 & 7 & 300\\1 & 8 & 230\\1 & 7 & 40\\1 & 9 & 0\\1 & 6 & 330\\1 & 9 & 180\end{matrix}\right]$，$X^{'}  =\left[\begin{matrix}1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\10 & 8 & 8 & 5 & 7 & 8 & 7 & 9 & 6 & 9\\80 & 0 & 200 & 200 & 300 & 230 & 40 & 0 & 330 & 180\end{matrix}\right]$ is the transpose of $X$, also called $X^{T} ,X^{tr} $,etc.,$y=\left[\begin{matrix}469\\366\\371\\208\\246\\297\\363\\436\\198\\364\end{matrix}\right]$。For a matrix $X$, the inverse matrix is $X^{-1} $. Using matrix evaluation methods, it is still available using the Sympy library, which provides a matrix building and matrix evaluation function. The final way to solve a multiple regression equation is to use the `sklearn.linear_model.LinearRegression` calculation directly and to obtain a regression model.

> The partial regression coefficient is a special property of multiple regression problems. Set up the independent variables $x_{1}, x_{2}, \ldots ,x_{m}, $, which has a linear relationship with the dependent variable $y$. there are $y= a_{1} x_{1} + a_{2} x_{2}+ \ldots + a_{n} x_{n}+b$，then $a_{1} , a_{2} , \ldots , a_{n} $ partial coefficients relative to each independent variables, represents the increase or decrease of the dependent variable y for each unit change in a given independent variable when all other variables remain constant.

* The matrix representation of the parameters solution formula $\widehat{ \beta } = ( X^{'} X)^{-1} X^{'}Y$ to understand:

Multiple linear regression model formula: $y=  \alpha  + \beta _{1} x_{1} + \beta _{2} x_{2}+  \ldots + \beta _{n} x_{n} $，can be simply expressed as:$Y=X \beta $，Its matrix is expressed as follows:：$\begin{bmatrix} Y_{1}   \\Y_{2}\\ \vdots\\Y_{n}  \end{bmatrix} = \begin{bmatrix} \alpha + \beta  X_{1}   \\\alpha + \beta  X_{2}\\ \vdots \\\alpha + \beta  X_{n}  \end{bmatrix} = \begin{bmatrix}1 &  X_{1}  \\1 &  X_{2} \\ \vdots & \vdots \\1 &  X_{n}  \end{bmatrix}  \times  \begin{bmatrix} \alpha  \\ \beta  \end{bmatrix} $，Since the matrix cannot be divided, you cannot directly divide both sides of $Y=X \beta $ by $X$ to find $\beta$. But you can multiply both sides by the inverse of $\beta$ to avoid division(The matrix times its inverse gives 1). And only a square matrix can be invertible, and you cannot control the number of samples, so $X$ times its transpose to get a square matrix that can be invertible.

```python
import sympy,math
from sympy import diff,Eq,solveset,solve,simplify,pprint,Matrix

a1,a2,b=sympy.symbols('a1 a2 b')
#Calculate the residual sum of squares
storeInfo_df["ss_res"]=storeInfo_df.apply(lambda row:(row.monthly_turnover-(row.area*a1+row.distance_to_nearestStation*a2+b))**2,axis=1)
util.print_html(storeInfo_df,10)
SS_res=storeInfo_df["ss_res"].sum()

#A- The Sympy library solved multiple regression equations.
#Differentiate the residual sum of squares 'SS_res' to a1, a2, and b, and set the differential value as 0.
diff_SSres_a1=diff(SS_res,a1)
diff_SSres_a2=diff(SS_res,a2)
diff_SSres_b=diff(SS_res,b)

#When the differential value is 0, solve the equation set and obtain a1, a2, and b.
Eq_residual_a1=Eq(diff_SSres_a1,0) #Make the differential value a1 as 0
Eq_residual_a2=Eq(diff_SSres_a2,0) #Make the differential value a2 as 0
Eq_residual_b=Eq(diff_SSres_b,0) #Make the differential value b as 0
slop_intercept=solve((Eq_residual_a1,Eq_residual_a2,Eq_residual_b),(a1,a2,b)) #Solve three-variable linear equation
print("diff_a1,a2 and intercept:\n")
pprint(slop_intercept)
print("_"*50)

#B - Multiple regression equations were solved using the matrix (based on Sympy)
if 'one' not in storeInfo_df.columns:
    X_m=Matrix(storeInfo_df.insert(loc=1,column='one',value=1)[['one','area','distance_to_nearestStation']])
else:
    X_m=Matrix(storeInfo_df[['one','area','distance_to_nearestStation']])
y_m=Matrix(storeInfo_df.monthly_turnover)

parameters_reg=(X_m.T*X_m)**-1*X_m.T*y_m #Note that matrix multiplication does not change position arbitrarily when calculating matrices.
print("matrix_a1,a2 and intercept:\n")
pprint(parameters_reg)

#C - Use Sklearn to solve multiple regression equations.
#B - Use sklearn library: sklearn.linear_model.LinearRegression()，Ordinary least squares Linear Regression-Obtain the regression equation by ordinary least squares linear regression
from sklearn.linear_model import LinearRegression
X=storeInfo_df[['area','distance_to_nearestStation']].to_numpy()
y=storeInfo_df['monthly_turnover'].to_numpy()

#Fitting model
LR_multivariate=LinearRegression().fit(X,y)
#Mode parameters
print("_"*50)
print("Sklearn a1=%.2f,a2=%.2f,intercept=%.2f"%(LR_multivariate.coef_[0],LR_multivariate.coef_[1], LR_multivariate.intercept_))

#Establish the regression equation
x1,x2=sympy.symbols('x1,x2')
fx_m=slop_intercept[a1]*x1+slop_intercept[a2]*x2+slop_intercept[b]
print("linear regression_fx=:\n")
pprint(fx_m)
fx_m=sympy.lambdify([x1,x2],fx_m,"numpy")
```

    diff_a1,a2 and intercept:
    
    ⎧    4073344      -44597      6409648⎫
    ⎨a₁: ───────, a₂: ───────, b: ───────⎬
    ⎩     98121        130828      98121 ⎭
    __________________________________________________
    matrix_a1,a2 and intercept:
    
    ⎡6409648⎤
    ⎢───────⎥
    ⎢ 98121 ⎥
    ⎢       ⎥
    ⎢4073344⎥
    ⎢───────⎥
    ⎢ 98121 ⎥
    ⎢       ⎥
    ⎢-44597 ⎥
    ⎢───────⎥
    ⎣ 130828⎦
    __________________________________________________
    Sklearn a1=41.51,a2=-0.34,intercept=65.32
    linear regression_fx=:
    
    4073344⋅x₁   44597⋅x₂   6409648
    ────────── - ──────── + ───────
      98121       130828     98121 
    


```python
#You can print the matrix as a mathematical expression in Latex format for easy presentation in markdown without typing it yourself.
from sympy import latex
print(latex(X_m.T))
```

    \left[\begin{matrix}1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1\\10 & 8 & 8 & 5 & 7 & 8 & 7 & 9 & 6 & 9\\80 & 0 & 200 & 200 & 300 & 230 & 40 & 0 & 330 & 180\end{matrix}\right]
    

Also, use a ternary diagram to print a chart between the two independent variables and the predicted values to see the variables' relationship.


```python
pd.options.mode.chained_assignment = None
storeInfo_df['pre']=LR_multivariate.predict(X)
columns=['area','distance_to_nearestStation','monthly_turnover','pre']
storeInfo_plot=storeInfo_df[columns]
normalize_df=storeInfo_plot.T.apply(lambda row:(row-row.min())/(row.max()-row.min()) , axis=1).T
normalize_df["location"]=storeInfo_df.location

import plotly.express as px
fig = px.scatter_ternary(normalize_df, a="pre", b="area",c="distance_to_nearestStation",hover_name="location",
                         color="area",size="pre", size_max=8,
                         ) 

fig.show()
```
<a href=""><img src="./imgs/7_9.png" height="auto" width="auto" title="caDesign"></a>

#### 1.3.4 Confirm the accuracy of multiple regression equation
The determination coefficient with a non-modified degree of freedom is similar to the simple linear regression. Place the defined calculation function `coefficient_of_determination` in the file 'util.py' and call it directly. Meanwhile, 'r2_score' provided by Sklearn was also used for calculation, and the calculation result was about 0.94, indicating that the index of the degree of fitting between the measured value and the predicted value of the regression equation is relatively high, which can better predict the monthly turnover according to the store area and the nearest station distance.


```python
#Calculate the multiple correlation coefficient R
import util
R_square_a,R_square_b=util.coefficient_of_determination(storeInfo_df.monthly_turnover.to_list(),storeInfo_df.pre.to_list())   
print("R_square_a=%.5f,R_square_b=%.5f"%(R_square_a,R_square_b))

from sklearn.metrics import r2_score
R_square_=r2_score(storeInfo_df.monthly_turnover.to_list(),storeInfo_df.pre.to_list())
print("using sklearn libray to calculate r2_score=",R_square_)
```

    R_square_a=0.94524,R_square_b=0.94524
    using sklearn libray to calculate r2_score= 0.945235852681711
    

* The non-modified degree determination coefficient 
When the determination coefficient is directly used, the more independent variables it has, the higher the determination coefficient's value. However, not every independent variable is effective, so the modified degree determination coefficient is usually used, and its formula is as follows:$R^{2} =1- \frac{  \frac{SS_{res}}{ n_{s} - n_{v} -1}  }{  \frac{SS_{tot}}{n_{s} -1}  }$，$n_{s}$ is the number of samples, $n_{s}$ is the number of the independent variables, $SS_{res}$ is the residual sum of squares, $SS_{tot}$ is the total sum of squared residuals.


```python
def coefficient_of_determination_correction(observed_vals,predicted_vals,independent_variable_n):
    import pandas as pd
    import numpy as np
    import math
    '''
    function - The non-modified degree determination coefficient of the regression equation
    
    Paras:
    observed_vals - Observed value(measured value)
    predicted_vals - Predicted value
    independent_variable_n - the number of the independent variable
    '''
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    #The deviation sum of squares of the observed values (the total sum of squares, or the total sum of squares of deviations) 
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    
    #The residual sum of squares
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
    
    #Determination coefficient
    sample_n=len(observed_vals)
    R_square_correction=1-(SS_res/(sample_n-independent_variable_n-1))/(SS_tot/(sample_n-1))
            
    return R_square_correction
R_square_correction=coefficient_of_determination_correction(storeInfo_df.monthly_turnover.to_list(),storeInfo_df.pre.to_list(),2)
print("The modified degree determination coefficient =",R_square_correction)
```

    The modified degree determination coefficient= 0.929588953447914
    

#### 1.3.5 Regression significance test
The regression coefficient test in a simple regression model, only need the given $H_{0} :A=0, H_{1} :A \neq 0$, but in multiple regression in terms of the population is $F_{x} = A_{1} x_{1} +A_{2} x_{2}+B$, $A_{1} \sim a_{1} ,A_{2} \sim a_{2},B \sim b$, $\sim$ is approximate. Including $A_{1}$ and $A_{2}$ two partial correlation coefficient, thus it can be divided into two kinds of circumstances; one is a comprehensive discussion of the partial regression coefficient, the original hypothesis is $A_{1} =A_{2}=0$, the alternative hypothesis is that $A_{1} =A_{2}=0$  is invalid, or any of the following set of relations is valid, $A_{1} \neq 0$ and $A_{2} \neq 0$, $A_{1} \neq 0$ and $A_{2} = 0$, or $A_{1} =0$ and $A_{2} \neq 0$. The other is to discuss the test of the partial regression coefficient separately, such as the null hypothesis is $A_{1} =0$, and the alternative hypothesis is $A_{1} \neq 0$. In these two ways, the test statistics are different, the statistics are $F_{0}= \frac{ SS_{tot}- SS_{res} }{ n_{v} } / \frac{SS_{res} }{ n_{s}- n_{v}-1} $, $SS_{tot}$ is the total sum of squares, $SS_{res}$ is the residual sum of squares, $n_{s}$ is the number of samples, $n_{v}$ is the number of the independent variables. For the test of a single regression coefficient, its statistics is $F_{0}= \frac{ a_{1} ^{2} }{ C_{jj} } / \frac{ SS_{res} }{ n_{s}- n_{v} -1 } $, $ C_{jj}$ is the diagonal value of $( X^{'} X)^{-1} $ at the intersection, that is, $( X^{'} X)^{-1} =\left[\begin{matrix}\frac{511351}{98121} & - \frac{55781}{98121} & - \frac{1539}{327070}\\- \frac{55781}{98121} & \frac{6442}{98121} & \frac{66}{163535}\\- \frac{1539}{327070} & \frac{66}{163535} & \frac{67}{6541400}\end{matrix}\right]$, $\frac{6442}{98121} $ is the diagonal value.

For the whole regression coefficient test and the single regression coefficient test, the P-value is less than 0.05, which means that the multiple linear regression model is appropriate.


```python
def ANOVA_multivarialbe(observed_vals,predicted_vals,independent_variable_n,a_i,X):
    import pandas as pd
    import numpy as np
    import math
    from scipy.stats import f
    from sympy import Matrix,pprint
    '''
    function - Multiple linear regression equation - regression significance test(regression coefficient test), the population test of all regression coefficient, and single regression coefficient tests
    
    Paras:    
    observed_vals - Observed value(measured value)
    predicted_vals - Predicted value
    independent_variable_n - The number of the independent variables
    a_i - Partial correlation coefficient list
    X - Sample dataset_the independent variable
    '''
    vals_df=pd.DataFrame({'obs':observed_vals,'pre':predicted_vals})
    #The total sum of squares, or the total sum of the squares deviation
    obs_mean=vals_df.obs.mean()
    SS_tot=vals_df.obs.apply(lambda row:(row-obs_mean)**2).sum()
    
    #The residual sum of squares
    SS_res=vals_df.apply(lambda row:(row.obs-row.pre)**2,axis=1).sum()
   
    #The regression sum of squares
    SS_reg=vals_df.pre.apply(lambda row:(row-obs_mean)**2).sum()
    
    #The number of samples
    n_s=len(observed_vals)
    dfn=independent_variable_n
    dfd=n_s-independent_variable_n-1
    
    #Calculate the total test statistics for all regression coefficients.
    F_total=((SS_tot-SS_res)/dfn)/(SS_res/dfd)
    print("F-distribution statistic_total=%.6f;p-value=%.6f"%(F_total,f.sf(F_total,dfn,dfd)))
    
    #Calculate the test statistics of a single regression coefficient one by one
    X=np.insert(X,0,1,1)
    X_m=Matrix(X)
    M_inverse=(X_m.T*X_m)**-1
    C_jj=M_inverse.row(1).col(1)[0]
    pprint(C_jj)
    
    F_ai_list=[]
    i=0
    for a in a_i:
        F_ai=(a**2/C_jj)/(SS_res/dfd)
        F_ai_list.append(F_ai)
        print("a%d=%.6f，F-distribution statistic_=%.6f;p-value=%.6f"%(i,a,F_ai,f.sf(F_total,1,dfd)))
        i+=1
 
a1_,a2_=LR_multivariate.coef_[0],LR_multivariate.coef_[1]
X=storeInfo_df[['area','distance_to_nearestStation']].to_numpy()
ANOVA_multivarialbe(storeInfo_df.monthly_turnover.to_list(),storeInfo_df.pre.to_list(),2,a_i=[a1_,a2_],X=X) 
```

    F-distribution statistic_total=60.410426;p-value=0.000038
     6442
    ─────
    98121
    a0=41.513478，F-distribution statistic_=44.032010;p-value=0.000110
    a1=-0.340883，F-distribution statistic_=0.002969;p-value=0.000110
    

#### 1.3.6 Population regression $A_{1}  X_{1} + A_{2}  X_{2}+ \ldots + A_{n}  X_{n}+B$ estimation——confidence interval
多元线性回归模型的预测值置信区间估计使用了两种计算方式，一是，自定义函数逐步计算，其计算公式为：$\sqrt{F(1,n_s-n_v-1;0.05) \times ( \frac{1}{n_s}+ \frac{ D ^{2} }{ n_s-1 }  ) \times   \frac{SS_{res}}{n_s-n_v-1}  } $，其中$n_s$为样本个数，$n_v$为自变量个位数，$D ^{2}$为马氏距离（Mahalanobis distance）的平方，$SS_{res}$为残差平方和；$D ^{2}$马氏距离的平方计算公式为：先求$S=\begin{bmatrix} S_{11} &S_{12} & \ldots &S_{1p}  \\S_{21}  &S_{22}& \ldots &S_{2p}\\ \vdots & \vdots & \ddots & \vdots \\ S_{p1} &S_{p2}& \ldots &S_{pp}   \end{bmatrix} $的逆矩阵$S^{-1} $，其中，$S_{22}$代表第2个自变量的离差平方和，$S_{25}$代表第2个自变量和第5个自变量的离差积和，$S_{25}$与$S_{52}$是相等的，以此类推；然后根据$S^{-1}$，求取马氏距离的平方公式为：$D^{2} =[( x_{1}- \overline{ x_{1} }  )( x_{1}- \overline{ x_{1} }) S^{11} +( x_{1}- \overline{ x_{1} }  )( x_{2}- \overline{ x_{2} }) S^{12}]+ \ldots +( x_{1}- \overline{ x_{1} }  )( x_{p}- \overline{ x_{p} }) S^{1p}\\+( x_{2}- \overline{ x_{2} }  )( x_{1}- \overline{ x_{1} }) S^{21} +( x_{2}- \overline{ x_{2} }  )( x_{2}- \overline{ x_{2} }) S^{12}]+ \ldots +( x_{2}- \overline{ x_{2} }  )( x_{p}- \overline{ x_{p} }) S^{2p}\\ \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots  \ldots \\+( x_{p}- \overline{ x_{p} }  )( x_{1}- \overline{ x_{1} }) S^{p1} +( x_{p}- \overline{ x_{p} }  )( x_{2}- \overline{ x_{2} }) S^{12}]+ \ldots +( x_{p}- \overline{ x_{p} }  )( x_{p}- \overline{ x_{p} }) S^{pp}(n_s-1)$，其中$n_s$为样本个数。

二是，使用[statsmodels](https://www.statsmodels.org/stable/index.html)的`statsmodels.regression.linear_model.OLS`普通最小二乘法（Ordinary Least Squares，OLS）求得多元线性回归方程，其语法结构与Sklearn基本相同。所求的的回归模型包含有置信区间的属性，可以通过`dt=res.get_prediction(X).summary_frame(alpha=0.05)`的方式提取。可以打印statsmodels计算所得回归模型的概要（summary），比较求解回归方程的偏回归系数和截距（coef_const/area/distance_to_nearestStation ），以及确认多元回归方程的精度R-squared（$R^2$）和修正自由度的判定系数Adj. R-squared，和回归显著性检验全面讨论偏回归系数的检验F-分布统计量F-statistic，对应P值Prob (F-statistic)，全部相等，互相印证了所使用的方法是否保持一致。

对于两种方法在预测变量置信区间比较上，分别打印了各自的三维分布图，其结果显示二者的图形保持一致，即通过statsmodels求解多元回归方程与逐步计算所得结果保持一致。

> [statsmodels](https://www.statsmodels.org/stable/index.html) 提供了一些类和函数，用于估计许多不同的统计模型，以及执行统计测试和统计数据研究。每个估计器都有一个广泛的结果统计信息列表，可以用以查看相关信息，以确保所求得的估计器（模型）的准确性、正确性。


* 马氏距离（Mahalanobis distance）

马氏距离表示数据的协方差矩阵，有效计算两个未知样本集相似度的方法。与欧式距离（Euclidean distance）不同的是它考虑到各种特性之间的联系（例如身高和体重是由关联的），并且是尺度无关的（scale-invariant，例如去掉单位）,独立于测量尺度。计算公式如上所述，也可以简化表示为，对于一个均值为$ \vec{ \mu }= (  \mu _{1}, \mu _{2}, \mu _{3}, \ldots , \mu _{N} )^{T} $（即为各个自变量的均值）的多变量（多个自变量）的矩阵，$ \vec{ x }= (  x_{1}, x _{2}, x _{3}, \ldots , x _{N} )^{T}$，其马氏距离为$D_{M} (\vec{ x })= \sqrt{ (\vec{ x }-\vec{ \mu })^{T} S^{-1}  (\vec{ x }-\vec{ \mu })} $。


```python
import numpy as np
import statsmodels.api as sm

#使用statsmodels库求解回归方程，与获得预测值的置信区间
storeInfo_df_sort=storeInfo_df.sort_values(by=['area'])
X=storeInfo_df_sort[['area','distance_to_nearestStation']]
X=sm.add_constant(X) #因为在上述逐步计算或者使用Sklearn求解回归方程过程中，多元回归方程均增加了常量截距的参数，因此此处增加一个常量 adding a constant
y=storeInfo_df_sort['monthly_turnover']
mod=sm.OLS(y,X) #构建最小二乘模型 Describe model
res=mod.fit() #拟合模型 Fit model
print(res.summary())   # Summarize model

dt=res.get_prediction(X).summary_frame(alpha=0.05)
y_prd = dt['mean']
yprd_ci_lower = dt['obs_ci_lower']
yprd_ci_upper = dt['obs_ci_upper']
ym_ci_lower = dt['mean_ci_lower'] 
ym_ci_upper = dt['mean_ci_upper']

#逐步计算
def confidenceInterval_estimator_LR_multivariable(X,y,model,confidence=0.05):
    import pandas as pd
    from sympy import Matrix,pprint
    import numpy as np
    '''
    function - 多元线性回归置信区间估计，以及预测区间
    
    Paras:
    X - 样本自变量 DataFrame数据格式
    y - 样本因变量
    model - 多元回归模型
    confidence - 置信度
    
    return:
    CI- 预测值的置信区间
    '''
    #根据指定数目，划分列表的函数
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    X_deepCopy=X.copy(deep=True) #如果不进行深度拷贝，如果传入的参数变量X发生了改变，则该函数外部的变量值也会发生改变
    columns=X_deepCopy.columns
    n_v=len(columns)
    n_s=len(y)
    
    #求S，用于马氏距离的计算
    SD=[]
    SD_name=[]
    for col_i in columns:
        i=0
        for col_j in columns:
            SD_column_name=col_i+'S'+str(i)
            SD_name.append(SD_column_name)
            if col_i==col_j:
                X_deepCopy[SD_column_name]=X_deepCopy.apply(lambda row: (row[col_i]-X_deepCopy[col_j].mean())**2,axis=1)
                SD.append(X_deepCopy[SD_column_name].sum())
            else:
                X_deepCopy[SD_column_name]=X_deepCopy.apply(lambda row: (row[col_i]-X_deepCopy[col_i].mean())*(row[col_j]-X_deepCopy[col_j].mean()),axis=1)
                SD.append(X_deepCopy[SD_column_name].sum())                
            i+=1
    M=Matrix(list(chunks(SD,n_v)))
    
    #求S的逆矩阵
    M_invert=M**-1
    #pprint(M_invert)
    M_invert_list=[M_invert.row(row).col(col)[0] for row in range(n_v) for col in range(n_v)]
    X_mu=[X_deepCopy[col].mean() for col in columns]
    
    #求马氏距离的平方
    SD_array=X_deepCopy[SD_name].to_numpy()    
    D_square_list=[sum([x*y for x,y in zip(SD_selection,M_invert_list)])*(n_s-1) for SD_selection in SD_array]    
    
    #计算CI-预测值的置信区间
    print(columns)
    ss_res=(y-model.predict(X_deepCopy[columns]))**2
    SS_res=ss_res.sum()
    print(SS_res)
    probability_val=f.ppf(q=1-confidence,dfn=1, dfd=n_s-n_v-1) 
    CI=[math.sqrt(probability_val*(1/n_s+D_square/(n_s-1))*SS_res/(n_s-n_v-1)) for D_square in D_square_list]

    return CI

X_=storeInfo_df_sort[['area','distance_to_nearestStation']]
y_=storeInfo_df_sort['monthly_turnover']
CI=confidenceInterval_estimator_LR_multivariable(X_,y_,LR_multivariate,confidence=0.05)

#打印图表
fig, axs=plt.subplots(1,2,figsize=(25,11))
x_=X.area
y_=X.distance_to_nearestStation

#由自定义函数，逐步计算获得的置信区间
axs[0]=fig.add_subplot(1,2,1, projection='3d')
axs[0].plot(x_,y_, y, linestyle = "None", marker = "o",markerfacecolor = "None", color = "black",label = "actual")
X_array=X_.to_numpy()
LR_pre=LR_multivariate.predict(X_array)
axs[0].plot(x_, y_,LR_pre, color = "red",label = "prediction")
axs[0].plot(x_,y_, LR_pre+CI, color = "darkgreen", linestyle = "--", label = "Confidence Interval")
axs[0].plot(x_,y_, LR_pre-CI, color = "darkgreen", linestyle = "--")

#由statsmodels库计算所得的置信区间
axs[1]=fig.add_subplot(1,2,2, projection='3d')
axs[1].plot(x_,y_, y, linestyle = "None", marker = "o",markerfacecolor = "None", color = "black",label = "actual")
axs[1].plot(x_, y_,y_prd, color = "red",label = "OLS")
axs[1].plot(x_, y_,yprd_ci_lower, color = "blue", linestyle = "--",label = "Prediction Interval")
axs[1].plot(x_, y_,yprd_ci_upper, color = "blue", linestyle = "--")

axs[1].plot(x_,y_, ym_ci_lower, color = "darkgreen", linestyle = "--", label = "Confidence Interval")
axs[1].plot(x_,y_, ym_ci_upper, color = "darkgreen", linestyle = "--")

axs[1].view_init(210,250) #可以旋转图形的角度，方便观察
axs[1].set_xlabel('area')
axs[1].set_ylabel('distance_to_nearestStation')
axs[1].set_zlabel('confidence interval')

axs[0].legend()
axs[1].legend()
axs[0].view_init(210,250) #可以旋转图形的角度，方便观察
axs[1].view_init(210,250) #可以旋转图形的角度，方便观察
plt.show()
```

    C:\Users\richi\AppData\Roaming\Python\Python37\site-packages\scipy\stats\stats.py:1535: UserWarning:
    
    kurtosistest only valid for n>=20 ... continuing anyway, n=10
    
    

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:       monthly_turnover   R-squared:                       0.945
    Model:                            OLS   Adj. R-squared:                  0.930
    Method:                 Least Squares   F-statistic:                     60.41
    Date:                Sat, 25 Jul 2020   Prob (F-statistic):           3.84e-05
    Time:                        11:20:05   Log-Likelihood:                -44.358
    No. Observations:                  10   AIC:                             94.72
    Df Residuals:                       7   BIC:                             95.62
    Df Model:                           2                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================================
                                     coef    std err          t      P>|t|      [0.025      0.975]
    ----------------------------------------------------------------------------------------------
    const                         65.3239     55.738      1.172      0.280     -66.476     197.124
    area                          41.5135      6.256      6.636      0.000      26.720      56.307
    distance_to_nearestStation    -0.3409      0.078     -4.362      0.003      -0.526      -0.156
    ==============================================================================
    Omnibus:                        0.883   Durbin-Watson:                   3.440
    Prob(Omnibus):                  0.643   Jarque-Bera (JB):                0.448
    Skew:                           0.479   Prob(JB):                        0.799
    Kurtosis:                       2.603   Cond. No.                     1.40e+03
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.4e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    Index(['area', 'distance_to_nearestStation'], dtype='object')
    4173.006119994701
    


<a href=""><img src="./imgs/7_10.png" height="auto" width="auto" title="caDesign"></a>


### 1.4 要点
#### 1.4.1 数据处理技术

* 使用sympy库建立方程，求解方程组，以及微分、矩阵计算；使用sympy的pprint打印方程及变量

* 用机器学习库[Sklearn](https://scikit-learn.org/stable/)，以及[statsmodels](https://www.statsmodels.org/stable/index.html)求解回归方程，以及计算模型精度（判定系数）、回归模型的显著性检验。

* 用plotly库的`px.scatter_ternary`，或者matplotlib库的`projection='3d'`方式，表述有三个变量的关系

#### 1.4.2 新建立的函数

* function - 在matplotlib的子图中绘制连接线，`demo_con_style(a_coordi,b_coordi,ax,connectionstyle)`

* function - 回归方程的判定系数， `coefficient_of_determination(observed_vals,predicted_vals)`

* function - 简单线性回归方程-回归显著性检验（回归系数检验）， `ANOVA(observed_vals,predicted_vals,df_reg,df_res)`

* function - 简单线性回归置信区间估计，以及预测区间， `confidenceInterval_estimator_LR(x,sample_num,X,y,model,confidence=0.05)`

* function - DataFrame数据格式，成组计算pearsonr相关系数，`correlationAnalysis_multivarialbe(df)`

* function - 回归方程的修正自由度的判定系数， `coefficient_of_determination_correction(observed_vals,predicted_vals,independent_variable_n)`

* function - 多元线性回归方程-回归显著性检验（回归系数检验），全部回归系数的总体检验，以及单个回归系数的检验， `ANOVA_multivarialbe(observed_vals,predicted_vals,independent_variable_n,a_i,X)`

* function - 多元线性回归置信区间估计，以及预测区间， `confidenceInterval_estimator_LR_multivariable(X,y,model,confidence=0.05)`

#### 1.4.3 所调用的库


```python
import sympy
from sympy import Symbol
from sympy import diff
from sympy import init_printing,pprint,sqrt
from sympy import ln,log,Eq
from sympy import Matrix
from sympy import Eq,solveset,solve,simplify
from sympy import latex

import numpy as np
import seaborn as sns
import plotly.express as px

import matplotlib.pyplot as plt
from matplotlib import cm

import math
import pandas as pd

from scipy import stats
from scipy.stats import f
from scipy.stats import pearsonr

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.metrics import r2_score

import statsmodels.api as sm
```

#### 1.4.4 参考文献
1. [日]高桥 信著作,Inoue Iroha,株式会社 TREND-PRO漫画制作,张仲恒译.漫画统计学之回归分析[M].科学出版社.北京.2009.08；
2. Timothy C.Urdan.Statistics in Plain English(白话统计学)[M].中国人民大学出版社.2013,12.第3版.
3. Douglas C.Montgomery,Elizabeth A.Peck,G.Geoffrey Vining著.王辰勇译.线性回归分析导论(Introduction to linear regression analysis).机械工业出版社.2016.04(第5版)
