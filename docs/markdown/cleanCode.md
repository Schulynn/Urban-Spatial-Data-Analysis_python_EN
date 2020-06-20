# clean code
> If you can browse this section before starting the code, check if your code is clean after you have started writing. It is wisdom in itself.

Every piece of code should be an essay containing the author's problem-solving thinking logic. Even the readers who do not understand the code should be able to read it smoothly, to have a general understanding of what the author wants to solve, and how to solve it. Many language books, primarily introductory or bulky manuals, can take the reader on a detour if they do not mention the neat way to code, especially for beginners, so comb through some key points for reference.

* **The name of variables should reflect the meaning of the variables themselves.**

For the sake of convenience, when writing the code often do not pay attention to the naming of the variable name. Still, a random name, such as code line, is filled with 'y'、'i'、'j'、'k,' such a single-character name, and 'jh', 'ik' combination of letters do not reflect the variable meaning. After a few days, when you need to go back to apply the code or transfer the code to a new project, you cannot quickly understand the meaning of the variable. It means that the code is not readable, which will waste time to understand it again and generate obstacles for others to transfer and apply the code.

There are three suggested forms for variable name naming：

So, first, individual words, such as `markers=['.','+','o','^','x','s']`, is a list of types of markers that have been defined in a graph to print from the Matplotlib chart library, so that the name can reflect the meaning of the variable being described.。

The second is the letter combination. One is `landmarkPts` in `landmarkPts=[Point(coordi[0],coordi[1]) for coordi in np.stack((landmarks[0], landmarks[1]), axis=-1)]`, or do not introduce the abbreviation as `landmarkPoints`, which can be understood directly as landmarks, which takes the form of the first letter capitalized after the combination of letter. The second is `landmark_pts` or `landmark_points,` where multiple letters are separated by an underscore in the middle, and the first letter is not necessary to be capitalized.

* **Give keynotes**

The importance of note-taking should not be questioned. Although proper variable names can explain the meaning of statements to some extent, the logic of problem-solving usually requires notes (# or \``` note \```), especially some critical thinking. For example, notes on functions:
```python
'''
With the folder name as a key of a dictionary, the value is a list of all filenames under that folder. File types can be defined by themselves.
'''
def filePath(dirpath,fileType):
    fileInfo={}
    ...
```
And an explanation of critical statements：
```python
 idx = tempDf[ tempDf['cluster'] == -1 ].index  #Delete the row corresponding to the field 'cluster' value of -1, namely the independent OSM point data, and no clustering, is formed.
 tempDf.drop(idx,inplace=True)  
```
Without notes in the above example, you would not immediately have a mental picture of what the code statements are doing. You would have to spend a lot of time extrapolating and even reading the entire code.  Of course, putting a function or variable with a proper name first, followed by notes, rather than poor variable names, and piles of notes put the cart before the horse.

* **Try to keep a function defined to do only one thing, with minimal transfer costs**

Each piece of function code is a solution to a problem, and many times this method will be used repeatedly in the same project or different projects, so the function should be portable with minimal or no changes when the code is used immediately. In general, we will not notice that global variables are included in a sing function, and the transfer will prompt the situation that global variables are not defined. Therefore, it is recommended to try to pass variables through function parameters when defining functions. Name, at the same time, in a single function and variable in a function, try to keep the name of the generality and avoid the name only reflect the content of the original project after transfer.  For example, the function `ffill() of `def ffill(arr):` represents the empty data to be filled forward in the array. If the function name and parameter names are changed to the specific `def landmarks_ffill(landmarks_arr),` the transferred variables may not be landmarks, which may cause ambiguity and interference in code reading.

The best purpose for a function to do one thing is to keep the code readable and transferable. It is not hard to understand, but it gets more complicated if the function can do more than one thing at a time. It is especially hard to categorize, and things get complicated when other projects only need part of the functionality of the function.

* **Complex programs need to learn to use classes and file-splitting_system thinking**

Unconsciously, the code can be complicated, less than one thousand lines, more than ten thousand lines, then all code in a file, or contains dozens of separate functions, the management of the code is very problematic. So learn to organize function with relevant properties, as well as segment code into multiple files. For example, `import driverlessCityProject_spatialPointsPattern_association_basic as basic,` by importing and aliasing the file as `basic` to use the code, the structure of the code will be more explicit and avoid the disadvantage of not easy to find in the same file. 

Classes and sub-files are only a means of systematic thinking. Robustness is only ensured when it comes to the organization of all code in a project, after determining the flow direction of data, the configuration of the hierarchy before and after the project, and the overall structure.

> Robert C. Martin.<em>Clean Code: A Handbook of Agile Software Craftsmanship</em>[M].U.S. Prentice Hall.August 11,2008. this book is well worth reading. It describes the philosophy of code, whether just entered the code field, or have been immersed in code for many years, read through, and then occasionally double, will insist on coding to solve problems.