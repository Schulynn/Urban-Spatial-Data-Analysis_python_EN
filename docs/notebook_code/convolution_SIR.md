> Created on Sat Dec  9 10/27/03 2017  @author: Richie Bao-caDesign (cadesign.cn)
> __+updated on Thu Aug 27 09/23/03 2020   

## 1. Convolution, SIR propagation model,  cost raster and species dispersal, SIR spatial propagation model
Ecologists who study biology have found it useful to distinguish between where an organism lives and what it does. The habitat of an organism is the place where it lives. The habitat characteristics are described in terms of evident physical characteristics, which are often the dominant life forms of plants or animals. Examples include forest habitats, desert habitats, and coral habitats. Because the concept of habitat emphasizes biological exposure conditions, it provides a reference for protecting the biological environment and constructing habitat conditions. The niche represents the range of conditions that an organism can tolerate, their way of life, their role in the ecological system. Each species has a unique niche. The unique morphological and physiological characteristics of organisms determine their tolerance conditions, such as feeding and escaping from predators. No living thing is capable of surviving under any conditions on earth. Environmental conditions vary from place to place in terms of spatial variation, with variations in climate, topography, and soil types leading to large-scale heterogeneity(from meters to hundreds of kilometers). Heterogeneity at small scales is generally due to plant structure, animal activity, and soil inclusions. Specific spatial scales may be necessary to one animal, not to another. When an individual moves through the continually changing environment of space, the faster the individuals move, and the smaller the scale of space changes, the faster it will encounter new environmental conditions. Topography and geology can alter local environments in otherwise homogeneous climates.  For example, in mountain areas, slope inclination and sunlight affect soil temperature and humidity, which is also applicable to urban areas. The height and distribution of buildings form the diverse local microclimate environment of cities. 

Although plants have a relatively small selection of where they live, they still tend to grow in environments suitable for growth, such as high soil mineral concentrations. Animals, on their own, can move around their environment and choose their habitat. Even within the habitats, there are significant differences in temperature, humidity, salinity, and other factors. Moreover, it is possible to differentiate microhabitats or microenvironments further. For example, the shade from a shrub in a desert is cooler and moisture than in an area exposed to direct sunlight. As animals live in various and changing environments, they often need to take action to make decisions. Many of these decisions involve food, looking for food, how long to eat in particular plate habitats, and what type of food to eat. The optimal foraging theory explains the animals' behavior to obtain the most profit from a selection. For example, when the birds feed their offspring in the nest, they can search for food freely in the distance from the nest's center. Risk-sensitive feeding, depending on the rate at which an individual can search for food and the area's relative safety. Prey selection, the intrinsic value of each type of food based on its nutrition and energy, and the difficulty of dealing with toxins and the potential dangers; And mixing foods to meet the necessary nutrients.

Geospatial factors are involved, in addition to satisfying suitable habitats(niches) and biological action decisions, as well as individual movements that maintain spatial connections of populations. Population biologists call the movement of individuals within a population dispersal, which refers to the movement of individuals between subpopulations. In the planning field, the method often used to protect a species is habitat suitability assessment. Alternatively, calculate the cost raster to plan the urban layout, road line selection, and other contents by comprehensively considering the biology, water security, historical value, land price, landscape value, recreation value, and residential value. The above method is widely applied in fields such as planning, landscape architecture, ecological planning. Furthermore, [SIR（Susceptible Infected Recovered Model）](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SIR_model) model is a kind of communication model, typical applications in infectious disease transmission path simulation, including estimates how to spread, the total number of infections, duration of the epidemic, and related parameters.

Because the spatial resistance distribution involved in the SIR model is similar to the cost raster of suitability evaluation in planning, the spread of infection sources and viruses is similar to the migration of species in ecology. The SIR model is also one of the simplest compartmental models, many of which are derivatives of this basic form; therefore, further to expand the current research results of cost raster, an attempt was made to introduce it into the planning field.

In the implementation of the SIR model, the convolution method is used, and convolution is the basis of the convolutional neural network in deep learning and image processing or feature extraction. Both of these aspects are essential contents in urban spatial data analysis methods. Therefore, the implementation of convolution and its code is explained first.

> Reference
> 1. Robert E.Ricklefs.The economy of nature[M].W. H. Freeman; 6th Edition (December 17, 2008).  ------Very recommended reading materials(textbooks), illustrated with pictures.

### 1.1 Convolution
In functional analysis, convolution is a mathematical operator which generates the third function through two functions $f$ and $g$. If one function is regarded as an indicator of an interval(system response function, $g$) and the other as an input function to be processed(input signal function, $f$), convolution can be regarded as an extension of the "smoothing average." Let $f(x)$, $g(x)$ be two integrable functions on $\mathbb{R}$, take the integral: $\int_ { - \infty} ^ \infty f( \tau )g(x- \tau )d \tau $, and prove that integral exists for almost all $x \in (- \infty , \infty )$. As the different values of the $x$, this integral defines a new function $h(x)$, called $f$ and $g$ convolution, as $h(x)=(f*g)(x)$. To understand the above formula, it is necessary to have the knowledge of integration(refer to the elaboration of integration section for details). It should also be noted that, for the convenience of understanding, the indicator function and the input function are regarded as curve functions in a time segment. Fix $f( \tau )$ at the time point of $T$(the x-coordinate is time point), flip $g(\tau)$ as $g(-\tau)$, and align the time point of $f(\tau)$ by translation, and the result is $g(T-\tau)$.

The interpretation of convolution computation can be understood in the following diagram. A key point to pay attention to is that at time T, for example, when T=20, time 20 is the newly entered signal. The smaller the last moment is, the longer the signal enters. For example, at time 0, the signal has gone through 20-time points, and at time 1, it has gone through 19-time points. The indicator function acts on the new entry signal of the input function at time 0, that is, the value at time 20; at time 1, the indicator function acts on the value at time 19 of the input function, and so on. It is just a reverse process, so the operation such as flipping and moving the response function appears. If the indicator function is fixed, and the input function's value is moved in step by step over time, this will not happen and will be easier to understand.

> The main object of functional analysis is the function space composed of functions. Using the term functional as a representation, the representative use of functions of functions means that the functions' arguments are functions.


```python
import numpy as np
import sympy
from sympy import pprint,Piecewise
t,t_=sympy.symbols('t t_')

'''Define indicator function'''
e=sympy.E
g_t=-1*((e**t-e**(-t))/(e**t+e**(-t)))+1 #Refer to Hyperbolic tangent function which is the hyperbolic tangent function  y=tanh x
g_t_=sympy.lambdify(t,g_t,"numpy")

'''Define input function'''
f_t=1*sympy.sin(4*np.pi*0.05*t_)+2.5 #+np.random.randn(1)
f_t_=sympy.lambdify(t_,f_t,"numpy")

'''Define time period'''
T=20 #time point
t_bucket=np.linspace(0, T, T+1, endpoint=True)

'''Drawing graphics'''
#The original position of function
fig, axs=plt.subplots(1,3,figsize=(30,3))
axs[0].plot(t_bucket,g_t_(t_bucket),'o',c='r',label='g_t')
axs[0].plot(t_bucket,f_t_(t_bucket),'o',c='b',label='f_t')
axs[0].plot([t_bucket,np.flip(t_bucket)],[g_t_(t_bucket),np.flip(f_t_(t_bucket))],'--',c='gray')
axs[0].set_title('original position')

#Flip response function
axs[1].plot(-t_bucket,g_t_(t_bucket),'o',c='r',label='g_t')
axs[1].plot(t_bucket,f_t_(t_bucket),'o',c='b',label='f_t')
axs[1].plot([np.flip(-t_bucket),t_bucket],[np.flip(g_t_(t_bucket)),f_t_(t_bucket)],'--',c='gray')
axs[1].set_title('flip g(t)')

#move response function
axs[2].plot(-t_bucket+T,g_t_(t_bucket),'o',c='r',label='g_t')
axs[2].plot(t_bucket,f_t_(t_bucket),'o',c='b',label='f_t')
axs[2].plot([np.flip(-t_bucket+T),t_bucket],[np.flip(g_t_(t_bucket)),f_t_(t_bucket)],'--',c='gray')
axs[2].set_title('move g(t)')

axs[0].legend(loc='upper right', frameon=False)
axs[1].legend(loc='upper right', frameon=False)
axs[2].legend(loc='upper right', frameon=False)
plt.show()
```


<a href=""><img src="./imgs/13_01.png" height="auto" width="auto" title="caDesign"></a>


#### 1.1.1 One dimensional convolution and curve segmentation
A simple example of one-dimensional convolution is given: the result of one set of indicator functions is [1,1,1], and the result of one set of input functions is [0,0,1,1,1,1,1,0,0]. The convolution result is obtained utilizing the fixed input function(usually the fixed indicator function, but the sliding input function), the sliding indicator function, the quadrature at the corresponding position, and the final sum to facilitate the calculation. Through calculation using `np.convolve(mode='full')`, its results are also `array([0, 0, 1, 2, 3, 3, 3, 2, 1, 0, 0])`.

|moment/step| step-0  |step-1   | step-2  | step-3  |step-4|step-5|step-6|step-7|step-8|step-sum|
|---|---|---|---|---|---|---|---|---|---|---|
|t-0| 0 | 0  | 1  | 1  | 1  |1|1|0|0|*0*|
|t-1| __1__*0  | -*0  | -*1  | -*1  | -*1  |-*1|-*1|-*0|-*0|*0*|
|t-2| __1__*0  | __1__*0  | -*1  | -*1  | -*1  |-*1|-*1|-*0|-*0|*0*|
|t-3| __1__*0  | __1__*0  | __1__*1  | -*1  | -*1  |-*1|-*1|-*0|-*0|*1*|
|t-4| -*0  | __1__*0  | __1__*1  | __1__*1   | -*1  |-*1|-*1|-*0|-*0|*2*|
|t-5| -*0  | -*0  |  __1__*1 | __1__*1  | __1__*1  |-*1|-*1|-*0|-*0|*3*|
|t-6| -*0  | -*0  | -*1  | __1__*1  | __1__*1  |__1__*1|-*1|-*0|-*0|*3*|
|t-7|  -*0 | -*0  | -*1  | -*1  | __1__*1 |__1__*1|__1__*1|-*0|-*0|*3*|
|t-8|  -*0  | -*0 |-*1 | -*1 |-*1 |__1__*1|__1__*1|__1__*0|-*0|*2*|
|t-9|  -*0  | -*0 |-*1 | -*1 |-*1 |-*1|__1__*1|__1__*0|__1__*0|*1*|
|t-10|  -*0  | -*0 |-*1 | -*1 |-*1 |-*1|-*1|__1__*0|__1__*0|*0*|
|t-11|  -*0  | -*0 |-*1 | -*1 |-*1 |-*1|-*1|-*0|__1__*0|*0*|
|t-12|  -*0  | -*0 |-*1 | -*1 |-*1 |-*1|-*1|-*0|-*0|*0*|

Based on the above examples, it is easier to understand convolution from the perspective of signal analysis. Hence, the name of the corresponding indicator function($g(x)$) is the system response function(or simply the response function), and the corresponding input function($f(x)$) is the input signal function(or simply the signal function). Define a class that outputs dynamic convolution given a response function and a signal function. In the process of the class definition, the animation method given in the Matplotlib library was used to define the dynamic diagram, and the animation of multiple subgraphs was realized by inheriting the `animation.TimedAnimation` parent class. Class configuration implements three graph functions: the first is the defined response function; usually a variable curve over a fixed period, applied to the signal function, enhanced, or the value of a specific variable signal function. The second sub-graph is a dynamic signal function. The signal function changes with time, which can be understood as the 'position' change of the signal at different times. For example, at time -4 in the figure, the signal is on the left side of the subgraph, i.e., it starts to enter, while at time 4, the signal is on the right side of the subgraph, i.e., it starts to go out. It can be interpreted as the signal passing over time. The third subgraph is the convolution of the calculation, and the response function is fixed. The entire scope is the valued interval, at every moment of the signal function at the specified time interval, so the convolution's position varies with the signal function but is not perfectly aligned.

The input parameters of the class include the 'G_T_fun' response function; 'F_T_fun' signal function;  and the beginning ('s') and end ('t') of 't' time. The 'step' of the time is used to configure the frame. The smaller the step, the smaller the time point update, and the higher the time accuracy. 'linespace' is the division of the time used for the x-axis coordinates of the chart and the signal's input parameters function and response function. The 'mode' parameter corresponds to `np.convolve(mode='')`, its modes includes 'same'，'full' and 'valid'. The definition of this class has not processed the 'valid' mode.

In the defined classed, the input parameters of 'F_T_fun', i.e., signal function, are specific in their definition, including the input parameter of a function, 'timing'(`f_t_sym=self.f_t_(i)`), to keep signals moving over time, applied to a Piecewise multisegment function supplied by Sympy library. And the input parameters(`f_t_val=f_t_sym(self.t)`) in Sympy formula defined in the function.


```python
%matplotlib inline
from IPython.display import HTML
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import sympy
from sympy import pprint,Piecewise

class dim1_convolution_SubplotAnimation(animation.TimedAnimation):
    '''
    function - One dimensional convolution animation analysis to customize the system function and signal function  
    
    Paras:
    G_T_fun - System response function
    F_T_fun - Input signal function
    t={"s":-10,"e":10,'step':1,'linespace':1000} -  Time start-point, end-point, frame step, time division
    mode='same' - The Numpy library provides 'convolve' convolution method
    '''    
    def __init__(self,G_T_fun,F_T_fun,t={"s":-10,"e":10,'step':1,'linespace':1000},mode='same'):  
        self.mode=mode
        self.start=t['s']
        self.end=t['e']
        self.step=t['step']
        self.linespace=t['linespace']
        self.t=np.linspace(self.start, self.end, self.linespace, endpoint=True)
        
        fig, axs=plt.subplots(1,3,figsize=(24,3))
        #define g(t)，system response function
        self.g_t_=G_T_fun        
        g_t_val=self.g_t_(self.t)
        self.g_t_graph,=axs[0].plot(self.t,g_t_val,'--',label='g(t)',color='r')        
        
        #define f(t)，input signal function
        self.f_t_=F_T_fun
        self.f_t_graph,=axs[1].plot([],[],'-',label='f(t)',color='b')
        
        #Convolution(dynamic - changing with time)
        self.convolution_graph,=axs[2].plot([],[],'-',label='1D convolution',color='g')

        axs[0].set_title('g_t')
        axs[0].legend(loc='lower left', frameon=False)
        axs[0].set_xlim(self.start,self.end)
        axs[0].set_ylim(-1.2,1.2)                
        
        axs[1].set_title('f_t')
        axs[1].legend(loc='lower left', frameon=False)
        axs[1].set_xlim(self.start,self.end)
        axs[1].set_ylim(-1.2,1.2)
        
        axs[2].set_title('1D convolution')
        axs[2].legend(loc='lower left', frameon=False)
        axs[2].set_xlim(self.start,self.end)
        axs[2].set_ylim(-1.2*100,1.2*100)

        plt.tight_layout()
        animation.TimedAnimation.__init__(self, fig, interval=500, blit=True) #interval configure update speed   
           
    #Update the figure
    def _draw_frame(self,framedata):    
        import math
        i=framedata
      
        f_t_sym=self.f_t_(i) #1-Start by entering the input parameters of the externally defined 'F_T_fun' function.
        f_t_val=f_t_sym(self.t) #2-Input parameters for a formula defined  in Sympy within the 'F_T_fun' function
        
        self.f_t_graph.set_data(self.t,f_t_val)    
        
        g_t_val=self.g_t_(self.t)
        g_t_val=g_t_val[~np.isnan(g_t_val)] #Removes null values, leaving only the data for the convolution portion
        
        if self.mode=='same':           
            conv=np.convolve(g_t_val,f_t_val,'same') #self.g_t_(t)
            self.convolution_graph.set_data(self.t,conv)            
            
        elif self.mode=='full':
            conv_=np.convolve(g_t_val,f_t_val,'full') #self.g_t_(t)
            t_diff=math.ceil((len(conv_)-len(self.t))/2)
            conv=conv_[t_diff:-t_diff+1]
            self.convolution_graph.set_data(self.t,conv)
            
        else:
            print("please define the mode value--'full' or 'same' ")
        
    #configure frames    
    def new_frame_seq(self):
        return iter(np.arange(self.start,self.end,self.step))
    
    #Initialize the figure
    def _init_draw(self):     
        graphs=[self.f_t_graph,self.convolution_graph,]
        for G in graphs:
            G.set_data([],[])        
```

* A simple response function and signal function definition implementation

The response function is a function of time length 1 and value 1. The signal function is a function whose value is 1 in a period with a length of 1 after the given time point('timing'), and the value of other times is 0(In effect, a continuous signal is generated at each point in time).  The response function is applied to the signal function, and the result is the triangle slope that the value first increases several times and then falls back.


```python
def G_T_type_1():
    import sympy
    from sympy import pprint,Piecewise
    '''
    function - Define system response functions. Type-1
    
    return:
    g_t_ - sympy defined functions
    '''
    t,t_=sympy.symbols('t t_')
    g_t=1
    g_t_piecewise=Piecewise((g_t,(t>=0)&(t<=1))) #A bit-piecewise function is defined. The system response function acts between the interval [0,1].
    g_t_=sympy.lambdify(t,g_t_piecewise,"numpy")
    #print("g_t=")
    #pprint(g_t)
    return g_t_

def F_T_type_1(timing):
    import sympy
    from sympy import pprint,Piecewise
    '''
    function - Define the input signal function. Type-1

    return:
    Function calculation formula
    '''
    t,t_=sympy.symbols('t t_')
    f_t=1
    f_t_piecewise=Piecewise((f_t,(t>timing)&(t<timing+1)),(0,True)) #A bit-piecewise function is defined. The system response function acts between the interval [0,1].
    f_t_=sympy.lambdify(t,f_t_piecewise,"numpy")
    #print("f_t=")
    #pprint(f_t)
    return f_t_   

G_T_fun=G_T_type_1() #system response function
F_T_fun=F_T_type_1 #input signal function
t={"s":-5,"e":5,'step':0.1,'linespace':1000,'frame':1} #Time parameter configuration
ani=dim1_convolution_SubplotAnimation(G_T_fun,F_T_fun,t=t,mode='same') #mode:'full','same'
HTML(ani.to_html5_video())  #conda install -c conda-forge ffmpeg  
```


<a href=""><img src="./imgs/13_02.gif" height="auto" width="auto" title="caDesign"></a>

```python
#Save the animation as a .gif file
from matplotlib.animation import FuncAnimation, PillowWriter 
writer=PillowWriter(fps=25) 
ani.save(r"./imgs/convolution_a.gif", writer=writer)
```

* A slightly more complex response function and signal function

Here, with a segment of the tanh hyperbolic tangent function as the response function, the value rapidly declines to a gradual decline. The signal function refers to a function whose value is 1 at a given time interval and 0 at other times. The convolution result produces a specific curvature of radian in ascending and descending sections(convex climbing, concave falling).


```python
def G_T_type_2():
    import sympy
    from sympy import pprint,Piecewise
    '''
    function - Define system response functions. Type-2
    
    return:
    g_t_ - sympy defined functions
    '''
    t,t_=sympy.symbols('t t_')
    e=sympy.E
    g_t=-1*((e**t-e**(-t))/(e**t+e**(-t)))+1 #refer to Hyperbolic tangent function  y=tanh x
    g_t_piecewise=Piecewise((g_t,(t>=0)&(t<3))) #A bit-piecewise function is defined. The system response function acts between the interval [0,3].
    g_t_=sympy.lambdify(t,g_t_piecewise,"numpy")
    
    #print("g_t=")
    #pprint(g_t)
    return g_t_

def F_T_type_2(timing):
    import sympy
    from sympy import pprint,Piecewise
    '''
    function - Define the input signal function. Type-2

    return:
    Function calculation formula
    '''
    t,t_=sympy.symbols('t t_')
    f_t=1
    f_t_piecewise=Piecewise((f_t,(t>timing)&(t<timing+1)) ,(f_t,(t>timing-2)&(t<timing-1)) ,(0,True)) #A bit-piecewise function is defined. The system response function acts between the interval [0,1].
    f_t_=sympy.lambdify(t,f_t_piecewise,"numpy")
    #print("f_t=")
    #pprint(f_t)
    return f_t_   

G_T_fun=G_T_type_2() 
F_T_fun=F_T_type_2
t={"s":-5,"e":5,'step':0.1,'linespace':1000,'frame':1} #Time parameter configuration
ani=dim1_convolution_SubplotAnimation(G_T_fun,F_T_fun,t=t,mode='same') #mode:'full','same'
HTML(ani.to_html5_video())  #conda install -c conda-forge ffmpeg   
```

```python
#Save the animation as a .gif file
from matplotlib.animation import FuncAnimation, PillowWriter 
writer=PillowWriter(fps=25) 
ani.save(r"./imgs/convolution_b.gif", writer=writer)
```

<a href=""><img src="./imgs/13_03.gif" height="auto" width="auto" title="caDesign"></a>


* Recognition and segmentation based on jump points of 1-d convolution curve

Given the convolution kernel(value of the convolution kernel), the input function can be influenced. In other words, if a convolution kernel(value of convolution kernel) is given and applied to a one-dimensional array(value of the input function), the input data can be changed to achieve the expected result. For example, the method of one-dimensional convolution can be used to segment the curve by jumping points. The convolution kernel used here is $[-1,2,-1]$. In this experimental analysis, the data used the PHMI simulated vehicle-mounted lidar navigation evaluation of the Driverless City Project(IIT), when $PHMI> 10^{-5} $, indicating that the feature points extracted from the lidar scanning point cloud can be used for good navigation fo the unmanned-vehicle; otherwise there will be risks. The experiment's input data can also generate a one-dimensional array(a list of values) at random. Since Matlab initially completed PHMI calculation, a Matlab chart file was generated to save the corresponding chart as .fig. This data can be loaded with the tools provided by the scipy.io library to extract the corresponding value.

> Note that the graph file of Matlab .fig, because of different operating and different Matlab versions, the code of data extraction may be adjusted accordingly. The .fig data extraction method cannot be extracted for all types of .fig files. Other extraction method functions are also defined in the driverless city project section.


```python
data_PHMI_fp=r'./data/04-10-2020_312LM_PHMI.fig'

def read_MatLabFig_type_A(matLabFig_fp,plot=True):
    from scipy.io import loadmat
    import matplotlib.pyplot as plt
    '''
    function - Read Matlab chart data, Type-A
    
    Paras:
    matLabFig_fp - MatLab chart data file path
    
    return:
    fig_dic - Return the chart data,（X,Y,Z）
    '''
    matlab_fig=loadmat(matLabFig_fp, squeeze_me=True, struct_as_record=False)
    fig_dic={} #.fig value of MatLab was extracted.
    ax1=[c for c in matlab_fig['hgS_070000'].children if c.type == 'axes']
    if(len(ax1) > 0):
        ax1 = ax1[0]
    i=0
    for line in ax1.children:
        try:
            X=line.properties.XData #good   
            Y=line.properties.YData 
            Z=line.properties.ZData
            fig_dic[i]=(X,Y,Z)
        except:
            pass     
        i+=1
        
    if plot==True:
        fig=plt.figure(figsize=(130,20))
        markers=['.','+','o','','','']
        colors=['#7f7f7f','#d62728','#1f77b4','','','']
        linewidths=[2,10,10,0,0,0]
        
        plt.plot(fig_dic[1][1],fig_dic[1][2],marker=markers[0], color=colors[0],linewidth=linewidths[0])  
        
        plt.tick_params(axis='both',labelsize=40)
        plt.show()
    
    return fig_dic

PHMI_dic=read_MatLabFig_type_A(matLabFig_fp=data_PHMI_fp)
```


<a href=""><img src="./imgs/13_04.png" height="auto" width="auto" title="caDesign"></a>


The basic idea of dividing the curve from the jump point is to compute the convolution to obtain the feature reflecting the jump point's change.->The convolution value is standardized by the standard score to facilitate the setting of the threshold. -> Returns the index value that satisfies the threshold. -> Define the function of dividing the list according to the index and divide it according to the index value meeting the threshold. -> Returns the split result, and the chart prints the result.


```python
def curve_segmentation_1DConvolution(data,threshold=1):
    import numpy as np
    from scipy import stats
    '''
    function - One dimensional convolution is applied to segment the data according to the jump point.
    
    Paras:
    data - One dimensional data to be processed.
    
    return:
    data_seg - List segmentation dictionary,"dataIdx_jump"-Split index value, "data_jump"-Split raw data, "conv_jump"-Split convolution result
    '''
    def lst_index_split(lst, args):
        '''
        function - Split the list according to the index
        
        transfer:https://codereview.stackexchange.com/questions/47868/splitting-a-list-by-indexes/47877 
        '''
        if args:
            args=(0,) + tuple(data+1 for data in args) + (len(lst)+1,)
        seg_list=[]
        for start, end in zip(args, args[1:]):
            seg_list.append(lst[start:end])
        return seg_list
    
    data=data.tolist()
    kernel_conv=[-1,2,-1] #Define the convolution kernel, the indicator function
    result_conv=np.convolve(data,kernel_conv,'same')
    #Standardized, convenient to determine the threshold,segmentation list according to the threshold 
    z=np.abs(stats.zscore(result_conv)) #Standard score - absolute value
    z_=stats.zscore(result_conv) #Standard score
    
    threshold=threshold
    breakPts=np.where(z > threshold) #Returns the index value that satisfies the threshold
    breakPts_=np.where(z_ < -threshold)
    
    #The list is segmented according to the index value meets the threshold
    conv_jump=lst_index_split(result_conv.tolist(), breakPts_[0].tolist()) #Split convolution result
    data_jump=lst_index_split(data, breakPts_[0].tolist()) #Split raw data
    dataIdx_jump=lst_index_split(list(range(len(data))), breakPts_[0].tolist()) #Split index value
    data_seg={"dataIdx_jump":dataIdx_jump,"data_jump":data_jump,"conv_jump":conv_jump}
    
    return data_seg

p_X=PHMI_dic[1][1]
p_Y=PHMI_dic[1][2]    
p_Y_seg=curve_segmentation_1DConvolution(p_Y)

'''Flattening list function'''
flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]

#Print segmentation result
import matplotlib.pyplot as plt
plt.figure(figsize=(130, 20))
plt.scatter(p_X, [abs(v) for v in flatten_lst(p_Y_seg["conv_jump"])],s=1) 

def nestedlst_insert(nestedlst):
    '''
    function - Nested list, sublists interpolated before and after
    
    Paras:
    nestedlst - Nested list
    
    return:
    nestedlst - The split list
    '''
    for idx in range(len(nestedlst)-1):
        nestedlst[idx+1].insert(0,nestedlst[idx][-1])
    nestedlst.insert(0,nestedlst[0])
    return nestedlst

def uniqueish_color():
    import matplotlib.pyplot as plt
    import numpy as np
    '''
    function - Use the method provided by Matplotlib to return floating-point RGB randomly
    '''
    return plt.cm.gist_ncar(np.random.random())

data_jump=p_Y_seg["data_jump"]
data_jump=nestedlst_insert(data_jump)

dataIdx_jump=p_Y_seg['dataIdx_jump']
dataIdx_jump=nestedlst_insert(dataIdx_jump)

p_X_seg=[[p_X[idx] for idx in g] for g in dataIdx_jump]
for val,idx in zip(data_jump,p_X_seg):
    plt.plot(idx, val, color=uniqueish_color(),linewidth=5.0)

plt.ylim(0,1.1)
plt.tick_params(axis='both',labelsize=40)
plt.show()
```


<a href=""><img src="./imgs/13_05.png" height="auto" width="auto" title="caDesign"></a>


#### 1.1.2 2 D convolution and image feature extraction
One dimensional convolution is the sliding of the convolution kernel and the sum of the corresponding positions after quadrature. The convolution in two dimensions is the same, where the convolution kernel has dimension two, usually odd, sliding on the two-dimensional plane, and taking the quadrature at the corresponding position and then sum it. The method `scipy.signal import convolve2d` was provided by the Scipy library, which also has 'mode' parameters('full','valid','same') and boundary processing parameters('fill','wrap','symm'). In the following test code, an image is called in. In the RGB value of the image, G and B are both 0, but R's value is changed, and there are only two values, red(255) and black(0). The convolution kernel is given, the convolution is calculated, and the data changes before and after observation are made. It is confirmed that the convolution kernel slides and the corresponding position product are summed to replace the image value of the corresponding position in the center of the convolution kernel.
 

Similarly, using images in the fashion_mnis dataset to test the convolution, the edge detection convolution kernel can also detect the image changes' boundary position.

> The fashion_mnist dataset is a fundamental data set for deep learning and could replace the widely used, countless re-uses of the MNIST handwritten dataset that no longer seem fresh. See the deep learning section for details.


```python
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.morphology import square
from scipy.signal import convolve2d

img_12Pix_fp=r'./data/12mul12Pixel.bmp'
fig, axs=plt.subplots(1,4,figsize=(28,7))

#A - Subfigure 1, the original image, and the R-value in the RGB value
img_12Pix=io.imread(img_12Pix_fp)
struc_square=square(12)
axs[0].imshow(img_12Pix,cmap="Paired", vmin=0, vmax=12)
for i in range(struc_square.shape[0]):
    for j in range(struc_square.shape[1]):
        axs[0].text(j, i, img_12Pix[:,:,0][i,j], ha="center", va="center", color="w")
axs[0].set_title('original img')

#B -  Subfigure 2, two-dimensional convolution, the convolution kernel can detect the boundary
kernel_edge=np.array([[-1,-1,-1],
                      [-1,8,-1],
                      [-1,-1,-1]])  #Edge detection convolution kernel/filter, namely, the convolution kernel

img_12Pix_int32=img_12Pix[...,0].astype(np.int32) #Sometimes, the image's default value is 'int8', and if the data type is not modified, the convolve calculation results will be wrong.
print('Verify that the dimensions are the same:',img_12Pix_int32.shape,kernel_edge.shape) #Just calculate the R-value
img_12Pix_convolve2d=convolve2d(img_12Pix_int32,kernel_edge,mode='same')
axs[1].imshow(img_12Pix_convolve2d)
for i in range(struc_square.shape[0]):
    for j in range(struc_square.shape[1]):
        axs[1].text(j, i, img_12Pix_convolve2d[i,j], ha="center", va="center", color="w")
axs[1].set_title('2d convolution')
   
#C- Using images in the fashion_mnist dataset, experimental convolution
from tensorflow import keras   
fashion_mnist=keras.datasets.fashion_mnist    
(train_images, train_labels), (test_images, test_labels)=fashion_mnist.load_data() 
fashion_mnist_img=train_images[900] #Extract an image randomly
axs[2].imshow(fashion_mnist_img)
axs[2].set_title('fashion_mnist_img')

#D-  Convolution of a random image in the fashion_mnist dataset
fashion_mnist_convolve2d=convolve2d(fashion_mnist_img,kernel_edge,mode='same')
axs[3].imshow(fashion_mnist_convolve2d)
axs[3].set_title('fashion_mnist_convolve2d')

plt.show()
```

    Verify that the dimensions are the same: (12, 12) (3, 3)
    


<a href=""><img src="./imgs/13_06.png" height="auto" width="auto" title="caDesign"></a>


Two-dimensional convolution is widely used in image processing to achieve particular effects or to extract image features. [The convolution kernel](https://lodev.org/cgtutor/filtering.html) used in image processing is listed below.

1. identity：$\begin{bmatrix}0 & 0&0 \\0 & 1&0\\0 & 0&0 \end{bmatrix} $

2. sharpen：$\begin{bmatrix}0 & -1&0 \\-1 & 5&-1\\0 & -1&0 \end{bmatrix} $，$\begin{bmatrix}-1 & -1&-1 \\-1 & 9&-1\\-1 & -1&-1 \end{bmatrix} $，$\begin{bmatrix}1 & 1&1 \\1 & -7&1\\1 & 1&1 \end{bmatrix} $，$\begin{bmatrix}-k & -k&-k \\-k & 8k+1&-k\\-k & -k&-k \end{bmatrix} $，$\begin{bmatrix}-1&-1&-1&-1&-1 \\-1&2&2&2&-1 \\-1&2&8&2&-1 \\-1&2&2&2&-1 \\1&-1&-1&-1&-1  \end{bmatrix} $

3. edge detection：$\begin{bmatrix}-1 & -1&-1 \\-1 & 8&-1\\-1 & -1&-1 \end{bmatrix}$， $\begin{bmatrix}0 & 1&0 \\1 & -4&1\\0 & 1&0 \end{bmatrix}$， $\begin{bmatrix}0&0&0&0&0 \\0&0&0&0&0  \\-1&-1&2&0&0  \\0&0&0&0&0  \\0&0&0&0&0   \end{bmatrix}$， $\begin{bmatrix}0&0&-1&0&0 \\0&0&-1&0&0  \\0&0&4&0&0  \\0&0&-1&0&0  \\0&0&-1&0&0   \end{bmatrix}$， $\begin{bmatrix}-1&0&0&0&0 \\0&-2&0&0&0  \\0&0&6&0&0  \\0&0&0&-2&0  \\0&0&0&0&-1   \end{bmatrix}$

4. embossing filter:$\begin{bmatrix}-1 & -1&0 \\-1 & 0&1\\0 & 1&1 \end{bmatrix}$ ，$\begin{bmatrix}-1&-1&-1&-1&0 \\-1&-1&-1&0&1 \\-1&-1&0&1&1  \\-1&0&1&1&1  \\0&1&1&1&1   \end{bmatrix}$

5. blur：box filter, averaging $\begin{bmatrix}1 & 1&1 \\1 & 1&1\\1 & 1&1 \end{bmatrix} \times  \frac{1}{9} $， $\begin{bmatrix}0. & 0.2&0. \\0.2 & 0.&0.2\\0. & 0.2&0. \end{bmatrix} $， $\begin{bmatrix}0&0&1&0&0 \\0&1&1&1&0  \\1&1&1&1&1  \\0&1&1&1&0 \\0&0&1&0&0  \end{bmatrix}$；gaussian blur,approximation$\begin{bmatrix}1 & 2&1 \\2 & 4&2\\1 & 2&1 \end{bmatrix} \times  \frac{1}{16} $；motion blur$\begin{bmatrix}1&0&0&0&0&0&0&0&0 \\0&1&0&0&0&0&0&0&0 \\0&0&1&0&0&0&0&0&0  \\0&0&0&1&0&0&0&0&0 \\0&0&0&0&1&0&0&0&0\\0&0&0&0&0&1&0&0&0\\0&0&0&0&0&0&1&0&0\\0&0&0&0&0&0&0&1&0\\0&0&0&0&0&0&0&0&1\end{bmatrix}$

### 1.2 SIR propagation model
To start the establishment of mathematical models, determine independent variables and dependent variables. The independent variable is time $t$, in days. Consider two sets of related dependent variables.

The first group of dependent variables calculates the number of people in each group(categorization), all as a function of time. $\begin{cases}S=S(t) &susceptible number \\I=I(t) &infected  number\\R=R(t) & recovered number\end{cases} $. The second group of dependent variables represents each proportion of the three types in the total population, assuming that $N$ is the total population. $\begin{cases}s(t)=S(t)/N &proportion of susceptible people\\i(t)=I(t)/N & proportion of infected people\\r(t)=R(t)/N & proportion of recovered people\end{cases} $，so $s(t)+i(t)+r(t)=1$. Using a total count may seem more natural, but using fractions instead makes the calculation easier. The two sets of dependent variables are proportional, so either set gives us the same information about how the virus spreads.

* Because factors that change the overall population, such as birth rates and migration, are ignored, infected individuals in the 'susceptible' move into the 'infected' people. In contrast, individuals in the 'infected' recover(or die) into the 'recovered' people. Assuming the rate of time change of $S(t)$, the number of people in the 'susceptible population' depends on the number of people already infected and the number of contacts between the susceptible and the infected. In particular, assuming that each infected person has a fixed number of contacts per day($b$), this is sufficient to transmit the disease. Simultaneously, not all contacts are 'susceptible.' Assuming that the population is evenly mixed and that the proportion of susceptible contacts is $s(t)$, on average, per infected person, produces $bs(t)$ newly infected person per day. (Since there are many 'susceptible' people and relatively few 'infected' people, some trickier calculation can be ignored, such as one susceptible individual encountering multiple susceptible individuals on the same day).

* It is also assumed that in any given day, a fixed percentage(k) of the infected population recovers(or dies) and enters the 'recovered population.'

**Differential equations for susceptible populations**,Based on the above assumptions, get $\frac{dS}{dt}=-bs(t)I(t) $. To facilitate understanding can assume that the total population $N=10$, under the current moment $t$, the number of susceptible is $S=4$($s=4/10$), the number of infected is $I=3$($i=3/10$), and the number of recovered is $R=3$($r=3/10$). Under the change time $\triangle t$, because the parameter $b$ is the fixed number of contacts of each person in the 'infected population,' assume $b=2$, then the contacts of the whole 'infected population' is $bI(t)=2 \times 3=6$. Since not all the contacts are 'susceptible population', assuming that the population is evenly mixed, the proportion of susceptible people in the contacts is $s(t)=4/10$, so $\triangle S=bI(t)s(t)=2 \times 3 \times 4/10=2.4$. Because the individuals in the 'susceptible population' have contacted the individuals in the 'infected population,' they changed to the new individuals in the 'infected population.' The number of individuals in the 'susceptible population' is reduced, so symbols are added. Finally, after the change time, $\triangle t$(without considering the conversion of I to R), $S=1.6，I=5.4，R=3$. According to the above differential equation, the overall count and fraction(or percentage) substitution is proportional; finally, the unit differential equation under the change of susceptible group $\triangle t$ is $\triangle t$. 


**Differential equations for recovered populations**，According to the above assumptions, $\frac{dr}{dt}=ki(t) $, where $k$ is the conversion ratio of 'infected population' to 'recovered population'. Applied the same assumption, if $k=0.2$, then $\triangle R=0.2*3=0.6$, in the end, $S=1.6，I=4.8,R=3.6$.

**Differential equations for infected populations**，Because $s(t)+i(t)+r(t)=1$, get $\frac{ds}{dt} + \frac{di}{dt} + \frac{dr}{dt} $, so $\frac{ds}{dt} + \frac{di}{dt} + \frac{dr}{dt} $.

The final SIR propagation model is：$\begin{cases}\frac{ds}{dt}=-bs(t)i(t) \\\frac{di}{dt} =bs(t)i(t)-ki(t)\\\frac{dr}{dt}=ki(t) \end{cases} $


> The explanatory reference for the SIR model comes from[MAA](https://www.maa.org/)(Mathematical Association of America)，[The SIR Model for Spread of Disease - The Differential Equation Model](https://www.maa.org/press/periodicals/loci/joma/the-sir-model-for-spread-of-disease-the-differential-equation-model)<Author：David Smith and Lang Moore>; Code reference [The SIR epidemic model](https://scipython.com/book/chapter-8-scipy/additional-examples/the-sir-epidemic-model/)


```python
import numpy as np
#Parameter configuration
N=1000 #Total population
I_0,R_0=1,0 #Initialize the affected population and recovered population
S_0=N-I_0-R_0 #The population of susceptible populations is calculated from the infected population and the recovered population.
beta,gamma=0.2,1./10 #Configure parameters b(beta) and k(gamma)
t=np.linspace(0,160,160) #Configure time series

#Define the SIR model, the differential function
def SIR_deriv(y,t,N,beta,gamma,plot=False):   
    import numpy as np
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt
    '''
    function - Define the SIR model, the differential function
    
    Paras:
    y - S,I,R Initialization values(for example, population)
    t - time series
    N - total population
    beta - The conversion rate from the susceptible population to the infected population
    gamma -The conversion rate from the infected population to the recovered population
    
    return:
    SIR_array - S, I, R number
    '''
    def deriv(y,t,N,beta,gamma):
        S,I,R=y
        dSdt=-beta*S*I/N
        dIdt=beta*S*I/N-gamma*I
        dRdt=gamma*I
        return dSdt,dIdt,dRdt
    
    deriv_integration=odeint(deriv,y,t,args=(N,beta,gamma))
    S,I,R=deriv_integration.T
    SIR_array=np.stack([S,I,R])
    #print(SIR_array)
    
    if plot==True:
        fig=plt.figure(facecolor='w',figsize=(12,6))
        ax=fig.add_subplot(111,facecolor='#dddddd',axisbelow=True)
        ax.plot(t,S/N,'b',alpha=0.5,lw=2,label='Susceptible')
        ax.plot(t,I/N,'r',alpha=0.5,lw=2,label='Infected')
        ax.plot(t,R/N,'g',alpha=0.5,lw=2,label='Recovered')
        
        ax.set_label('Time/days')
        ax.set_ylim(0,1.2)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=1, ls='-')
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)    
        plt.show()   
    
    return SIR_array
    

y_0=S_0,I_0,R_0
SIR_array=SIR_deriv(y_0,t,N,beta,gamma,plot=True)     
```


<a href=""><img src="./imgs/13_07.png" height="auto" width="auto" title="caDesign"></a>


### 1.3 Convolution diffusion, cost raster, and species dispersal
#### 1.3.1 Convolution diffusion
如果配置卷积核为$\begin{bmatrix}0.5 & 1&0.5 \\1 & -6&1\\0.5 & 1&0.5 \end{bmatrix} $，假设存在一个传播（传染）源，如下述程序配置一个栅格（.bmp图像），其值除了源为1（R值）外，其它的单元均为0。每次卷积，值都会以源为中心，向四周扩散，并且四角的值（绝对值）最小，水平垂直向边缘值（绝对值）稍大，越向内部值（绝对值）越大，形成了一个逐步扩散的趋势，并且具有强弱，即越趋向于传播源，其绝对值越大。同时，达到一定阶段后，扩散开始逐步消失，恢复为所有单元值为0的阶段。观察卷积扩散可以通过打印每次卷积后的值查看，也可以记录每次的结果最终存储为.gif文件，动态的观察图像的变化。


```python
def img_struc_show(img_fp,val='R',figsize=(7,7)):
    from skimage import io
    from skimage.morphology import square,rectangle
    import matplotlib.pyplot as plt
    '''
    function - 显示图像以及颜色R值，或G,B值
    
    Paras:
    img_fp - 输入图像文件路径
    val - 选择显示值，R，G，或B
    figsize - 配置图像显示大小
    '''
    img=io.imread(img_fp)
    shape=img.shape
    struc_square=rectangle(shape[0],shape[1])
    fig, ax=plt.subplots(figsize=figsize)
    ax.imshow(img,cmap="Paired", vmin=0, vmax=12)                     
    for i in range(struc_square.shape[0]):
        for j in range(struc_square.shape[1]):
            if val=='R':
                ax.text(j, i, img[:,:,0][i,j], ha="center", va="center", color="w")
            elif val=='G':
                ax.text(j, i, img[:,:,1][i,j], ha="center", va="center", color="w")                         
            elif val=='B':
                ax.text(j, i, img[:,:,2][i,j], ha="center", va="center", color="w")                         
    ax.set_title('structuring img elements')
    plt.show

img_12Pix_1red_fp=r'./data/12mul12Pixel_1red.bmp'
img_struc_show(img_fp=img_12Pix_1red_fp,val='R')
```


<a href=""><img src="./imgs/13_08.png" height="auto" width="auto" title="caDesign"></a>


扩散是一个随时间变化的动态过程，为了记录每一次扩散的结果，形成动态的变化显示，使用moviepy记录每次图像变化的结果，并存储为.gif文件，方便查看。卷积扩散主要是更新SIR变量值，其初始值配置为图像的R颜色值，除了源配置为1外，其它单元的值（R）均为0，这样可以比较清晰的观察卷积扩散的过程。卷积的计算使用scipy库提供的convolve方法。


```python
class convolution_diffusion_img:
    '''
    class - 定义基于SIR模型的二维卷积扩散
    
    Paras:
    img_path - 图像文件路径
    save_path - 保持的.gif文件路径
    hours_per_second - 扩散时间长度
    dt - 时间记录值，开始值
    fps - 配置moviepy，write_gif写入GIF每秒帧数
    '''
    def __init__(self,img_path,save_path,hours_per_second,dt,fps):
        from skimage import io
        import numpy as np   
        self.save_path=save_path
        self.hours_per_second=hours_per_second
        self.dt=dt
        self.fps=fps        
        img=io.imread(img_path)
        SIR=np.zeros((1,img.shape[0], img.shape[1]),dtype=np.int32) #在配置SIR数组时，为三维度，是为了对接后续物种散步程序对SIR的配置
        SIR[0]=img.T[0] #将图像的RGB中R通道值赋值给SIR
        self.world={'SIR':SIR,'t':0} 
        
        self.dispersion_kernel=np.array([[0.5, 1 , 0.5],
                                        [1  , -6, 1],
                                        [0.5, 1, 0.5]]) #SIR模型卷积核 
        

    '''返回每一步卷积的数据到VideoClip中'''
    def make_frame(self,t):
        while self.world['t']<self.hours_per_second*t:
            self.update(self.world) 
        if self.world['t']<6:
            print(self.world['SIR'][0])
        return self.world['SIR'][0]

    '''更新数组，即基于前一步卷积结果的每一步卷积'''
    def update(self,world):        
        disperse=self.dispersion(world['SIR'], self.dispersion_kernel)
        world['SIR']=disperse 
        world['t']+=dt  #记录时间，用于循环终止条件

    '''卷积扩散'''
    def dispersion(self,SIR,dispersion_kernel):
        import numpy as np
        from scipy.ndimage.filters import convolve
        return np.array([convolve(SIR[0],self.dispersion_kernel,mode='constant',cval=0.0)]) #注意卷积核与待卷积数组的维度
    
    '''执行程序'''
    def execute_(self):        
        import moviepy.editor as mpy
        self.animation=mpy.VideoClip(self.make_frame,duration=1) #duration=1
        self.animation.write_gif(self.save_path,self.fps)    
        

img_12Pix_fp=r'./data/12mul12Pixel_1red.bmp' #图像文件路径
SIRSave_fp=r'./data/12mul12Pixel_1red_SIR.gif'
hours_per_second=20
dt=1 #时间记录值，开始值
fps=15 # 配置moviepy，write_gif写入GIF每秒帧数
convDiff_img=convolution_diffusion_img(img_path=img_12Pix_fp,save_path=SIRSave_fp,hours_per_second=hours_per_second,dt=dt,fps=fps)
convDiff_img.execute_()
```

                                                       

    [[0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]]
    MoviePy - Building file ./data/12mul12Pixel_1red_SIR.gif with imageio.
    [[0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 1 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0]]
    [[  0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   1   0   0   0   0   0]
     [  0   0   0   0   0  -1 -11  -1   0   0   0   0]
     [  0   0   0   0   1 -11  40 -11   1   0   0   0]
     [  0   0   0   0   0  -1 -11  -1   0   0   0   0]
     [  0   0   0   0   0   0   1   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0]
     [  0   0   0   0   0   0   0   0   0   0   0   0]]
    [[   0    0    0    0    0    0    0    0    0    0    0    0]
     [   0    0    0    0    0    0    0    0    0    0    0    0]
     [   0    0    0    0    0    0    1    0    0    0    0    0]
     [   0    0    0    0    0   -5  -18   -5    0    0    0    0]
     [   0    0    0    0   -5    5   94    5   -5    0    0    0]
     [   0    0    0    1  -18   94 -286   94  -18    1    0    0]
     [   0    0    0    0   -5    5   94    5   -5    0    0    0]
     [   0    0    0    0    0   -5  -18   -5    0    0    0    0]
     [   0    0    0    0    0    0    1    0    0    0    0    0]
     [   0    0    0    0    0    0    0    0    0    0    0    0]
     [   0    0    0    0    0    0    0    0    0    0    0    0]
     [   0    0    0    0    0    0    0    0    0    0    0    0]]
    [[   0    0    0    0    0    0    0    0    0    0    0    0]
     [   0    0    0    0    0    0    1    0    0    0    0    0]
     [   0    0    0    0   -2  -13  -29  -13   -2    0    0    0]
     [   0    0    0   -2   -7   62  198   62   -7   -2    0    0]
     [   0    0    0  -13   62  -13 -769  -13   62  -13    0    0]
     [   0    0    1  -29  198 -769 2102 -769  198  -29    1    0]
     [   0    0    0  -13   62  -13 -769  -13   62  -13    0    0]
     [   0    0    0   -2   -7   62  198   62   -7   -2    0    0]
     [   0    0    0    0   -2  -13  -29  -13   -2    0    0    0]
     [   0    0    0    0    0    0    1    0    0    0    0    0]
     [   0    0    0    0    0    0    0    0    0    0    0    0]
     [   0    0    0    0    0    0    0    0    0    0    0    0]]
    

    

为了方便查看动态变化的.gif文件，定义`animated_gif_show`函数。对图像的读取和处理使用了[Pillow库](https://pillow.readthedocs.io/en/stable/)。


```python
def animated_gif_show(gif_fp,figsize=(8,8)):
    from PIL import Image, ImageSequence
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    '''
    function - 读入.gif，并动态显示
    
    Paras:
    gif_fp - GIF文件路径
    figsize - 图表大小
    '''
    gif=Image.open(gif_fp,'r')
    frames=[np.array(frame.getdata(),dtype=np.uint8).reshape(gif.size[0],gif.size[1]) for frame in ImageSequence.Iterator(gif)]

    fig=plt.figure(figsize=figsize)
    imgs=[(plt.imshow(img,animated=True),) for img in frames]

    anim=animation.ArtistAnimation(fig, imgs, interval=300,repeat_delay=3000, blit=True)
    return HTML(anim.to_html5_video())
    
gif_fp=r'./data/12mul12Pixel_1red_SIR.gif'
animated_gif_show(gif_fp,figsize=(8,8))
```





<a href=""><img src="./imgs/13_09.gif" height="auto" width="auto" title="caDesign"></a>


#### 1.3.2 成本栅格与物种散布，SIR的空间传播模型
SIR传播模型是给定了总体人口数N，以及S,I,R的初始值，和$\beta , \gamma$转换系数，计算S,I,R人口数量的变化。SIR传播模型不具有空间属性，因此引入了卷积扩散，可以实现一个源向四周扩散的变化过程，并且始终以源的空间位置强度最大，并向四周逐步减弱。这样通过结合SIR模型和卷积扩散，可以实现SIR的空间传播模型。

可以这样理解SIR空间传播模型的过程，对于空间分布（即栅格，每个栅格单元有一个值或多个值，即存在多层栅格）配置有三个对应的栅格层（'SIR'），分别对应S，I和R值的空间分布，例如对于S而言，其初始值对应了用地类型对于物种散步的影响，即成本栅格或者是空间阻力值，如物种容易在林地、农田中迁移散步，而建筑和道路则会阻挡其散步，通过配置不同大小的值可以反映物种能够散步的强度（注意，有时可以用大的值表示阻力值小，而小的值表示阻力值大，具体需要根据算法的方式确定，或调整为符合习惯的数值大小表达）；对于对应的I层栅格而言，初始值需要设定源，可以为一个或者多个，多个源可以是分散的，也可以是聚集为片区的，源也可以根据源自身的强度配置代表不同强度的值，但是除了源之外的所有值均为0；对于对应的R层，因为在开始时间点，并没有恢复者（或死亡）案例，因此所有值设置为0。

配置好SIR空间分布的栅格层后，计算可以理解为纵向的SIR模型传播，以及水平向的卷积扩散过程的结合。在纵向上，每一个空间点（即一个位置的单元栅格，每个位置有S，I和R３个层的栅格）都有一定数量的人口数（即空间阻力值），每一个位置对应的S，I和R单元栅格的SIR传播模型计算过程与上述解释的SIR模型传播过程是相同的，只是在一开始的时候，只有I层对应源的位置有非0值，因此源的栅格位置是有纵向（S-->I-->R）的传播，但是为0的位置，SIR模型计算公式变化结果为0，即没有纵向上的传播；除了纵向SIR传播，各层的水平向发生着卷积扩散，对于S ,I和R三个栅格层，水平向的扩散速度是不同的，通过变量'dispersion_rates'扩散系数确定强度，三个值分别对应S，I和R的3个栅格层。因为I层是给了值非0的源，因此会发生水平扩散，新扩散的区域在SIR模型计算时，纵向上就会发生传播变化。而在配置扩散系数时，对应的S层的扩散系数为0，即S栅格层水平向上并不会发生卷积扩散，卷积扩散仅发生在I和R层，这与S层为“易感人群”但并传播，而I层“受感人群”和R层“恢复人群或死亡”是病毒携带者是可以传染扩散的过程相一致。每一时间步，由纵向SIR传播和水平向卷积扩散共同作用，因此每一时刻的SIR空间栅格值变化为`world['SIR'] += dt*(deriv_infect+disperse)`，其中'deriv_infect'为纵向SIR的传播结果，'disperse'为水平向的扩散结果，通过求和作为二者共同作用的结果。同时，乘以'dt'时间参数，开始时，时间参数比较小，而随时间的流逝，时间参数的值越来越大，那么各层的绝对值大小也会增加的更快，即传播强度会增加。

同样应用moviepy库记录每一时刻的变化。

* 确定源的位置

首先读取分类数据，因为不同分类数据的标识各异，通常无法正常显示为颜色方便地物的辨别，因此建立'las_classi_colorName'分类值到颜色的映射字典，将其转换为不同颜色的表达，从而根据颜色辨别地物，并方便观察位置信息。位置信息的获取是使用skimage库提供的ImageViewer方法。

分类数据为点云数据处理部分的分类数据，因为SIR的空间传播模型计算中，卷积扩散计算过程比较耗时，因此需要压缩数据量，并确保分类值正确。压缩数据时，skimage.transform提供了rescale,resize，和downscale_local_mean等方法，但是没有提供给定一个区域（block），返回区域内频数最大的值的方式；skimage.measure中的block_reduce方法，仅有numpy.sum, numpy.min, numpy.max, numpy.mean 和 numpy.median返回值，因此自定义函数`downsampling_blockFreqency`，实现给定二维数组，和区域大小，以区域内出现次数最多的值为该区域的返回值，实现降采样。


```python
def downsampling_blockFreqency(array_2d,blocksize=[10,10]):
    import numpy as np
    from statistics import multimode
    from tqdm import tqdm
    #flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst] #展平列表函数
    '''
    fuction - 降采样二维数组，根据每一block内值得频数最大值，即最多出现得值为每一block的采样值
    
    Paras:
    array_2d - 待降采样的二维数组
    blocksize - block大小，即每一采用的范围
    '''
    
    shape=array_2d.shape
    row,col=blocksize
    row_p,row_overlap=divmod(shape[1],row)  #divmod(a,b)方法为除法取整，以及a对b的余数
    col_p,col_overlap=divmod(shape[0],col)
    print("row_num:",row_p,"col_num:",col_p)
    array_extraction=array_2d[:col_p*col,:row_p*row]  #移除多余部分，规范数组，使其正好切分均匀
    print("array extraction shape:",array_extraction.shape,"original array shape:",array_2d.shape)  
    
    h_splitArray=np.hsplit(array_extraction,row_p)
    
    v_splitArray=[np.vsplit(subArray,col_p) for subArray in h_splitArray]
    #mostFrequency=[[multimode(b.flatten())[0] for b in v] for v in v_splitArray]
    blockFrenq_list=[]
    for h in tqdm(v_splitArray):
        temp=[]
        for b in h:
            blockFrenq=multimode(b.flatten())[0]
            temp.append(blockFrenq)
        blockFrenq_list.append(temp)   
    downsample=np.array(blockFrenq_list).swapaxes(0,1)
    return downsample  

import sys
import skimage.io
import skimage.viewer
from skimage.transform import rescale
import pandas as pd
from matplotlib import colors
import numpy as np
from skimage.measure import block_reduce

classi_fp=r'F:\GitHubBigData\IIT_lidarPtClouds\mosaic\classification_mosaic.tif'
mosaic_classi_array = skimage.io.imread(classi_fp)
mosaic_classi_array_rescaled=downsampling_blockFreqency(mosaic_classi_array, blocksize=(20,20))

las_classi_colorName={0:'black',1:'white',2:'beige',3:'palegreen',4:'lime',5:'green',6:'tomato',7:'silver',8:'grey',9:'lightskyblue',10:'purple',11:'slategray',12:'grey',13:'cadetblue',14:'lightsteelblue',15:'brown',16:'indianred',17:'darkkhaki',18:'azure',9999:'white'}
las_classi_colorRGB=pd.DataFrame({key:colors.hex2color(colors.cnames[las_classi_colorName[key]]) for key in las_classi_colorName.keys()})
classi_array_color=[pd.DataFrame(mosaic_classi_array_rescaled).replace(las_classi_colorRGB.iloc[idx]).to_numpy() for idx in las_classi_colorRGB.index]
classi_img=np.stack(classi_array_color).swapaxes(0,2).swapaxes(0,1)
print("finished img preprocessing...")
```

    C:\Users\richi\anaconda3\envs\rasterio\lib\site-packages\skimage\viewer\utils\__init__.py:1: UserWarning: Recommended matplotlib backend is `Agg` for full skimage.viewer functionality.
      from .core import *
    

    row_num: 1250 col_num: 1125
    array extraction shape: (22500, 25000) original array shape: (22501, 25001)
    

    100%|██████████| 1250/1250 [01:31<00:00, 13.74it/s]
    

    finished img preprocessing...
    


```python
viewer=skimage.viewer.ImageViewer(classi_img)
viewer.show()
```




    []



<a href=""><img src="./imgs/13_10.png" height="auto" width="auto" title="caDesign"></a>

* 定义SIR的空间传播模型类


```python
class SIR_spatialPropagating:
    '''
    funciton - SIR的空间传播模型
    
    Paras:
    classi_array - 分类数据（.tif，或者其它图像类型），或者其它可用于成本计算的数据类型
    cost_mapping - 分类数据对应的成本值映射字典
    beta - beta值，确定S-->I的转换率
    gamma - gamma值，确定I-->R的转换率
    dispersion_rates - SIR三层栅格各自对应的卷积扩散率
    dt - 时间更新速度
    hours_per_second - 扩散时间长度/终止值(条件)
    duration - moviepy参数配置，持续时长
    fps - moviepy参数配置，每秒帧数
    SIR_gif_savePath - SIR空间传播计算结果.gif文件保存路径
    '''
    def __init__(self,classi_array,cost_mapping,start_pt=[10,10],beta=0.3,gamma=0.1,dispersion_rates=[0, 0.07, 0.03],dt=1.0,hours_per_second=7*24,duration=12,fps=15,SIR_gif_savePath=r'./SIR_sp.gif'):
        from sklearn.preprocessing import MinMaxScaler
                
        #将分类栅格，按照成本映射字典，转换为成本栅格(配置空间阻力)
        for idx,(identity,cost_value) in enumerate(cost_mapping.items()):
            classi_array[classi_array==cost_value[0]]=cost_value[1]
        self.mms=MinMaxScaler()
        normalize_costArray=self.mms.fit_transform(classi_array) #标准化成本栅格

        #配置SIR模型初始值，将S设置为空间阻力值
        SIR=np.zeros((3,normalize_costArray.shape[0], normalize_costArray.shape[1]),dtype=float)        
        SIR[0]=normalize_costArray
        
        #配置SIR模型中I的初始值。1，可以从设置的1个或多个点开始；2，可以将森林部分直接设置为I有值，而其它部分保持0。
        #start_pt=int(0.7*normalize_costArray.shape[0]), int(0.2*normalize_costArray.shape[1])  #根据行列拾取点位置
        #print("起始点:",start_pt)
        start_pt=start_pt
        SIR[1,start_pt[0],start_pt[1]]=0.8  #配置起始点位置值

        #配置转换系数，以及卷积核
        self.beta=beta #β值
        self.gamma=gamma #γ值
        self.dispersion_rates=dispersion_rates  #扩散系数
        dispersion_kernelA=np.array([[0.5, 1 , 0.5],
                                     [1  , -6, 1],
                                     [0.5, 1, 0.5]])  #卷积核_类型A    
        dispersion_kernelB=np.array([[0, 1 , 0],
                                     [1 ,1, 1],
                                     [0, 1, 0]])  #卷积核_类型B  

        self.dispersion_kernel=dispersion_kernelA #卷积核
        self.dt=dt  #时间记录值，开始值
        self.hours_per_second=hours_per_second  #终止值(条件) 
        self.world={'SIR':SIR,'t':0} #建立字典，方便数据更新
        
        #moviepy配置
        self.duration=duration
        self.fps=fps        
                 
        #保存路径
        self.SIR_gif_savePath=SIR_gif_savePath

    '''SIR模型'''
    def deriv(self,SIR,beta,gamma):
        S,I,R=SIR
        dSdt=-1*beta*I*S  
        dRdt=gamma*I
        dIdt=beta*I*S-gamma*I
        return np.array([dSdt, dIdt, dRdt])

    '''卷积扩散'''
    def dispersion(self,SIR,dispersion_kernel,dispersion_rates):
        from scipy.ndimage.filters import convolve
        return np.array([convolve(e,dispersion_kernel,cval=0)*r for (e,r) in zip(SIR,dispersion_rates)])

    '''执行SIR模型和卷积，更新world字典'''
    def update(self,world):
        deriv_infect=self.deriv(world['SIR'],self.beta,self.gamma)
        disperse=self.dispersion(world['SIR'], self.dispersion_kernel, self.dispersion_rates)
        world['SIR'] += self.dt*(deriv_infect+disperse)    
        world['t'] += self.dt

    '''将模拟计算的值转换到[0,255]RGB色域空间'''
    def world_to_npimage(self,world):
        coefs=np.array([2,20,25]).reshape((3,1,1))
        SIR_coefs=coefs*world['SIR']
        accentuated_world=255*SIR_coefs
        image=accentuated_world[::-1].swapaxes(0,2).swapaxes(0,1) #调整数组格式为用于图片显示的（x,y,3）形式
        return np.minimum(255, image)

    '''返回每一步的SIR和卷积综合蔓延结果'''
    def make_frame(self,t):
        while self.world['t']<self.hours_per_second*t:
            self.update(self.world)     
        return self.world_to_npimage(self.world)    

    '''执行程序'''
    def execute(self):
        import moviepy.editor as mpy
        animation=mpy.VideoClip(self.make_frame,duration=self.duration)  #12
        animation.write_gif(self.SIR_gif_savePath, fps=self.fps) #15
```

空间阻力值的配置，需要根据具体的研究对象做出调整。也可以增加新的多个条件，通过栅格计算后作为输入的一个条件栅格。`cost_mapping`成本映射字典，键为文字标识，值为一个元组，第一个值为分类值，第二个值为空间阻力值。


```python
#成本栅格（数组）
classi_array=mosaic_classi_array_rescaled    

#配置用地类型的成本值（空间阻力值）
cost_H=250
cost_M=125
cost_L=50
cost_Z=0
cost_mapping={
            'never classified':(0,cost_Z),
            'unassigned':(1,cost_Z),
            'ground':(2,cost_M),
            'low vegetation':(3,cost_H),
            'medium vegetation':(4,cost_H),
            'high vegetation':(5,cost_H),
            'building':(6,cost_Z),
            'low point':(7,cost_Z),
            'reserved':(8,cost_M),
            'water':(9,cost_M),
            'rail':(10,cost_L),
            'road surface':(11,cost_L),
            'reserved':(12,cost_M),
            'wire-guard(shield)':(13,cost_M),
            'wire-conductor(phase)':(14,cost_M),
            'transimission':(15,cost_M),
            'wire-structure connector(insulator)':(16,cost_M),
            'bridge deck':(17,cost_L),
            'high noise':(18,cost_Z),
            'null':(9999,cost_Z)       
            }    

import util
s_t=util.start_time()
#参数配置
start_pt=[418,640]  #[3724,3415]
beta=0.3
gamma=0.1
dispersion_rates=[0, 0.07, 0.03]  #S层卷积扩散为0，I层卷积扩散为0.07，R层卷积扩散为0.03
dt=1.0
hours_per_second=30*24 #7*24
duration=12 #12
fps=15 #15
SIR_gif_savePath=r'E:\dataset\results\SIR_sp.gif'

SIR_sp=SIR_spatialPropagating(classi_array=classi_array,cost_mapping=cost_mapping,start_pt=start_pt,beta=beta,gamma=gamma,dispersion_rates=dispersion_rates,dt=dt,hours_per_second=hours_per_second,duration=duration,fps=fps,SIR_gif_savePath=SIR_gif_savePath)
SIR_sp.execute()
util.duration(s_t)
```

    start time: 2020-09-05 23:00:45.507837
    

    t:   0%|          | 0/180 [00:00<?, ?it/s, now=None]

    pygame 1.9.6
    Hello from the pygame community. https://www.pygame.org/contribute.html
    MoviePy - Building file E:\dataset\results\SIR_sp.gif with imageio.
    

                                                                  

    end time: 2020-09-05 23:23:27.824231
    Total time spend:22.70 minutes
    

    

<a href=""><img src="./imgs/13_11.gif" height="auto" width="auto" title="caDesign"></a>

### 1.4 要点
#### 1.4.1 数据处理计数

* 使用sympy库的Piecewise方法建立分段函数

* 使用matplotlib库的animation方法实习动态图表

* 使用scipy库提供的odeint方法，组合多个常微分方程

#### 1.4.2 新建立的函数

* class - 一维卷积动画解析，可以自定义系统函数和信号函数，`class dim1_convolution_SubplotAnimation(animation.TimedAnimation)`

* function - 定义系统响应函数.类型-1， `G_T_type_1()`

* function - 定义输入信号函数，类型-1， `F_T_type_1(timing)`

* function - 定义系统响应函数.类型-2，`G_T_type_2()`

* function - 定义输入信号函数，类型-2， `F_T_type_2(timing)`

* function - 读取MatLab的图表数据，类型-A， `read_MatLabFig_type_A(matLabFig_fp,plot=True)`

* function - 应用一维卷积，根据跳变点分割数据， `curve_segmentation_1DConvolution(data,threshold=1)`

* function - 根据索引，分割列表，`lst_index_split(lst, args)`

* function - 展平列表函数， `flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]`

* function - 嵌套列表，子列表前后插值，`nestedlst_insert(nestedlst)`

* function - 使用matplotlib提供的方法随机返回浮点型RGB， `uniqueish_color()`

* function - 定义SIR传播模型微分方程， `SIR_deriv(y,t,N,beta,gamma,plot=False)`

* function - 显示图像以及颜色R值，或G,B值，`img_struc_show(img_fp,val='R',figsize=(7,7))`

* class - 定义基于SIR模型的二维卷积扩散，`class convolution_diffusion_img`

* function - 读入.gif，并动态显示，`animated_gif_show(gif_fp,figsize=(8,8))`

* fuction - 降采样二维数组，根据每一block内值得频数最大值，即最多出现得值为每一block的采样值，`downsampling_blockFreqency(array_2d,blocksize=[10,10])`

* class - SIR的空间传播模型， `SIR_spatialPropagating`

#### 1.4.3 所调用的库


```python
import numpy as np
import pandas as pd
import sys
import math

import sympy
from sympy import pprint,Piecewise

from IPython.display import HTML
import numpy as np

from scipy import signal
from scipy.io import loadmat
from scipy import stats
from scipy.signal import convolve2d
from scipy.integrate import odeint
from scipy.ndimage.filters import convolve

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from skimage import io
from skimage.morphology import square,rectangle
import skimage.io
import skimage.viewer
from skimage.transform import rescale
from skimage.measure import block_reduce

from tensorflow import keras  
import moviepy.editor as mpy
from statistics import multimode
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
```

#### 1.4.4 参考文献
1. Robert E.Ricklefs著. 孙儒泳等译.生态学/The economy of nature[M].高度教育出版社.北京.2004.7 第5版. ------非常值得推荐的读物(教材)，图文并茂
