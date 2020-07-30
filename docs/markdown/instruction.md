# instruction
## Focus on the stage of how to teach knowledge clearly
If the author has a little knowledge as only looks at the sky in a well, then he cannot explain the problem clearly in any way. Therefore, the first stage has spent a lot of time on experimental accumulation, and finally get a glimpse of the whole picture, and enters the second stage. In this stage, in addition to continuing to accumulate experiments, the focus is on how to teach knowledge. It is also the motivation for communicating with partners in an interactive community in the process of finishing the book rather than launching a community after finishing the book.

## Target audience
The title of 'Urban spatial data analysis method——python language implementation' reflects the principal direction of the research, such as urban planning, landscape architecture, architecture, ecological planning, and geographic information.  For readers, including college students and researchers, workgroup, are not impossible. The basic requirements of the python language require a minimum entry-level, and we try to solve this problem at this stage also, either as a separate warm-up or otherwise.

## How to learn
'How to learn?' is the biggest concern of readers, but also the author needs to solve it. We built the system's resource platform to meet the necessary data acquisition, code operation, interactive discussion to assist active learning. And the most critical is how to explain, which will continue to experiment in this stage, adjust to meet the requirements, and is also related to the feedback of partners in the community.

## Necessary knowledge supplement
In this book, because of the particularity of professional design/planning and discipline development, a lot of knowledge not configuration related course in university, so in the process of urban spatial data analysis, it involves a lot of knowledge points in combination with related literature into the experiment, to some extent, alleviates the burden of readers find relevant knowledge again. And from simple data, gradually to complex data analysis, we can understand the content of the study more clearly. With simple data sample 🍅 as identity,  return to the experimental data 🐨 as identity. 

## Learn by case study
A thousand words are not worth a simple case demonstration. Usually, see a specific function or method and the attribute's function, the best approach is to search case to view the data structure changes directly, often do not need to read the detailed explanation of the text, you can learn the function of this method from the changes of data, and then assist in understanding the text description, further find the explanation of unknown points.

## Take advantage of search engines
The world of the code is not marginal; it is impossible to remember all method functions. The code libraries are always updated and improved, and the knowledge you remember is not necessarily right. The key to learning code is still to get in the habit of coding, to train our thinking in coding, and ultimately to use this tool to solve real problems. Those countless libraries, functions, methods, properties, etc.  usually in the process of coding, make full use of the search engines to look up. One tends to find the answer platform [StackOverflow](https://stackoverflow.com/), the corresponding library website online document, etc. 。而经常用到的方法函数会在不知不觉中记住，或有意识的对经常用到的方法函数强化记忆，从而更快的记住，在下次用时将不用再搜索查找。

## 库的安装
Anaconda集成环境，使得python库的安装，尤其库之间的依赖关系不再是让coder们扰心的事情。在开始本书代码之前，不需要一气安装所有需要的库在一个环境之下，而是跟随着代码的编写，需要提示有什么库没有安装，再行安装最好。因为，虽然Anaconda提供了非常友好的库管理环境，但是还是会有个别库之间的冲突，在安装过程中也有可能造成失败，连同原以安装的库也会出现问题，从而无形之中增加了时间成本。同时，对于项目开发，最好仅建立支持本项目的环境，使之请便，网络环境部署时也不会有多余的部分。但是，对于自行练习或者不是很繁重的项目任务，或者项目的前期，只有环境能够支持不崩溃应该就可以。

## 电脑的配置
本书涉及到深度学习部分，为了增加运算速度，用到GPU，关于何种GPU可以查看相关信息。作者所使用的配置为(仅作参考)：
```
OMEN by HP Laptop 15
Processor - Intel(R) Core(TM) i9-9880H CPU @2.30GHz 
Installed RAM - 32.0GB(31.9GB usable)
System type - 64-bit operating system, x64-based processor

Display adapter - NVIDA GeForce RTX 2080 with Max-Q Design
```

> 建议内存最低32.0GB,最好64.0GB以上，甚至更高。如果内存较低，则考虑分批读取处理数据后合并等方法，善用内存管理，数据读写方式（例如用HDF5格式等）。

城市空间数据分析通常涉及到海量的数据处理，例如芝加哥城区域三维激光雷达数据约有1.41TB, 因此如果使用的是笔记本，最好配置容量大的外置硬盘，虽然会影响到读取和存储速度，但是避免笔记本自身存储空间的占用，影响运算速度。

## 善用`print()`
`print()`是python语言中使用最为频繁的语句，在代码编写、调试过程中，要不断的用该语句来查看数据的值、数据的变化、数据的结构、变量所代表的内容、监测程序进程、以及显示结果等等，`print()`实时查看数据反馈，才可知道代码编写的目的是否达到，并做出反馈，善用`print()`是用python写代码的基础。

## 库的学习
本书的内容是城市空间数据分析方法的python语言实现，在各类实验中使用有大量的相关库，例如[SciPy](https://www.scipy.org/)，[NumPy](https://numpy.org/)，[pandas](https://pandas.pydata.org/)，[scikit-learn](https://scikit-learn.org/stable/)，[PyTorch](https://pytorch.org/)，[statsmodels](https://www.statsmodels.org/stable/index.html)，[SymPy](https://www.sympy.org/en/index.html)，[Pygame](https://www.pygame.org/news)，[matplotlib](https://matplotlib.org/)，[Plotly](https://plotly.com/)，[seaborn](https://seaborn.pydata.org/)，[bokeh](https://docs.bokeh.org/en/latest/index.html)，[GeoPandas](https://geopandas.org/)，[GDAL](https://gdal.org/)，[rasterstats](https://pythonhosted.org/rasterstats/)，[Shapely](https://shapely.readthedocs.io/en/latest/manual.html)，[pathlib](https://docs.python.org/3/library/pathlib.html)，[PySAL](https://pysal.org/)，[NetworkX](https://networkx.github.io/)，[rasterio](https://rasterio.readthedocs.io/en/latest/)，[PDAL](https://pdal.io/)，[scikit-imge](https://scikit-image.org/)，[VTK](https://vtk.org/)，[flask](https://flask.palletsprojects.com/en/1.1.x/)，[sqlite3](https://docs.python.org/3/library/sqlite3.html)，[cv2](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html)，[re](https://docs.python.org/3/library/re.html)，[itertools](https://docs.python.org/3/library/itertools.html)，[urllib](https://docs.python.org/3/library/urllib.html)等，上述库为本书主要使用的库，很多库都是大体量，每个库单独学习都会花费一定时间。如果很多库之前没有接触过，一种方式是，在阅读到相关部分用到该库时，提前快速的学习各个库官方提供的教程(tutorial)，不是手册（manual），从而快速的掌握库的主要结构，再继续阅读本书实验；另一种方式是，为了不打断阅读的连续性，可以只查找用到该库的类或函数的内容。每个库的内容一般都会非常多，不需要一一学习，用到时根据需要有针对性，和有目的性的查阅即可。但是有些库是需要系统的看下其教程，例如scikit-learn、PyTorch等极其重量型的库。

## 避免重复造轮子
“避免重复造轮子”是程序员的一个‘座右铭’，当然每个人所持的观点不尽相同，但是可以肯定的是，是没有必要从0开始搭建所有项目内容的，这如同再写python的所有库，甚至python核心内容的东西，例如scikit-learn、PyTorch集成了大量的模型算法，我们只是直接使用，不会再重复写这些代码。但是在本书阐述过程中，会有意识的对最为基本的部分按照计算公式重写代码，目的是认识清楚这究竟是怎么回事，而真正的项目则无需如此。

## 从数学-公式到代码，图形：用Python语言学习数学的知识点
就像书面表达音乐的最佳方式是乐谱，那么最能巧妙展示数学特点的就是数学公式。很多人，尤其规划、建筑和景观专业，因为专业本身形象思维的特点，大学多年的图式思维训练，对逻辑思维有所抵触，实际上好的设计规划，除了具有一定的审美意识，空间设计能力，好的逻辑思维会让设计师在空间形式设计能力上亦有所提升，能够感觉到这种空间能力的变化，更别提，建筑大专业本身就是工科体系，是关乎到人们生命安全的严谨的事情。

当开始接触公式时，可能不适应，但是当你逐步的用公式代替所要阐述的方法时， 你会慢慢喜欢上这种方式，即使我们在小学就开始学习公式。

再者，很多时候公式，甚至文字内容让人费解，怎么也读不懂作者所阐述的究竟是什么，尤其包含大部分公式，而案例都是“白纸黑字”的论述时。这是因为我们只是看的到，但是摸不到，你很难去调整参数再实验作者的案例，但是在python等语言的世界中，数据、代码都是实实在在的东西，你可以用print()查看每一步数据的变化，尤其不易理解的地方，一比较前后的数据变化，和所用的什么数据，一切都会迎刃而解。这也是，为什么本书的所有阐述都是基于代码的，公式看不懂，没有问题；文字看不懂，也没有问题，我们只要逐行的运行代码，似乎就会明朗开来，而且尽可能的将数据以图示的方式表述，来增加理解的方式，让问题更容易浮出水面。所有有章节都有一个.ipynb的Jupyter文件，可以互动操作代码，实现的边学边学边实验的目的。

## 参数文献及书的推荐
城市空间数据分析方法需要多学科的融合，规划、建筑和景观的专业知识，python等语言知识，数学（统计学、微积分、线性代数、数据库等），以及地理信息系统，生态学等等大类，而其下则为更为繁杂的小类细分。实际上，我们只是用到在解决问题时所用到的知识，当在阅读时，如果该处的知识点不是很明白，可以查找相关文献来获取补充，在每一部分时，也提供了所参考的文献，书籍，有必要时可以翻看印证。

## 知识点的跳跃-没有方程式的推导
很多公式并不会引其推导过程，主要是专业目的性，一个专业做了所有专业的事，那是非常疯的，不现实，不实际，不可行，只有学科间不断的融合，才能快速的成长，在有必要就本专业研究问题寻找解决方法时，需要自行推导再用心做这件事，而已有的公式所代表的研究成果，例如概率密度函数公式、最小二乘法、判定系数等等，是不建议从头开始逐步推导。

## 对于那些绕弯的代码

At this point, because the book is being written, we can adapt flexibly, because there are environmental genes for learning together, rather than reading alone at night. This experience of growing together will resonate with the reader and author, and we believe that the interaction of the community will stimulate the interest and passion for learning.