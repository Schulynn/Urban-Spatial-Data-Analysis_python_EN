> Created on Sat Aug  1 22/20/30 2020 @author: Richie Bao-caDesign (cadesign.cn)

## 1. A code representation of the linear algebra basis
In the analysis of urban spatial data analysis methods, if you want to have a clear understanding of a certain method to solve the problem, a lot of mathematical knowledge cannot be avoided, and linear algebra is one of them. Basic knowledge of linear algebra is required for the matrix solution of regression equations, eigenvectors, dimensionality reduction(such as PCA), and space transformation. Therefore, the main ideas are based on the structure of *The Manga Guide to Linear Algebra*, combined with *Introduction to linear algebra*, and related references. Based on python, combined with python chart printing visualization, the main knowledge points are connected as the basic knowledge reserve of relevant chapters using linear algebra knowledge. The library mainly used is [SymPy(Matrices-linear algebra)](https://docs.sympy.org/latest/modules/matrices/matrices.html).

Linear algebra is a mathematical branch of vector spaces and linear mapping, including the study of lines, planes, subspaces, and all vector spaces' general properties. Generally speaking, it is a mathematics branch that connects the $m$ dimensional space(world) with $n$ dimensional space(world) using vector and matrix mathematical space expression. In fact, in the discipline of design and planning, the spatial transformation of many geometric objects using three-dimensional software platforms such as Rhinoceros(Grasshopper) used by us is the application of linear algebra.

> Reference for this part
> 1.  Shin Takahashi, Iroha Inoue.The Manga Guide to Linear Algebra.No Starch Press; 1st Edition (May 1, 2012);
> 2. Gilbert Strang.Introduction to linear algebra[M].Wellesley-Cambridge Press; Fifth Edition edition (June 10, 2016)

### 1.1 Matrix
Like $\begin{bmatrix} a_{11} & a_{12} &\ldots&a_{1n} \\ a_{21} & a_{22} &\ldots&a_{2n}\\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} &\ldots&a_{mn} \end{bmatrix} $ of dataset, which is $m$ rows, $n$ columns, $m \times n$ matrix, which is similar to Numpy array, can also be expressed in the DataFrame format. In statistics, the sample feature, especially those with multiple features, can be expressed by matrix. $m$ can be called a row tag, and $n$ can be called a column tag, and together they represent each element in the matrix. For like $\begin{bmatrix} a_{11} & a_{12} &\ldots&a_{1n} \\ a_{21} & a_{22} &\ldots&a_{2n}\\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} &\ldots&a_{nn} \end{bmatrix} $, in other words, the matrix of $m=n$ is the square matrix of $n$ order, and the elements at the diagonal position are called diagonal elements. In the correlation analysis, the array result of the calculated pairwise correlations is the square matrix, and the diagonal elements are all 1, which is the settlement result of their own correlation.




#### 1.1.1 Matrix operation

1. addition,subtraction 

The addition(subtraction) $A \pm B$ of $A$ and $B$ in the form of  matrix $m \times n$ is a $m \times n$ matrix, in which each elements is the addition(subtraction) of the corresponding elements in $A$ and $B$, $(A \pm B)_{i,j} = A_{i,j} \pm B_{i,j} $, for example  $\begin{bmatrix}1 & 3&1 \\1 & 0&0 \end{bmatrix} + \begin{bmatrix}0 & 0&5 \\7 & 5&0 \end{bmatrix} = \begin{bmatrix}1+0& 3+0&1+5 \\1+7 & 0+5&0+0 \end{bmatrix} = \begin{bmatrix}1 & 3 &6\\8 & 5&0 \end{bmatrix}  $

2. Scalar multiplication of matrix

The scalar $c$ is multiplied by the matrix $A$, each element of cA is the product of the corresponding element of $A$ and $c$,$(cA)_{i,j} =c  \cdot A_{i,j}  $，for example $2 \cdot \begin{bmatrix}1 & 3&1 \\1 & 0&0 \end{bmatrix}=\begin{bmatrix}2 \cdot1 & 2 \cdot3&2 \cdot1 \\2 \cdot1 & 2 \cdot0&2 \cdot0 \end{bmatrix}=\begin{bmatrix}2 & 6&2 \\2 & 0&0 \end{bmatrix}$

3. Multiplication

The multiplication of two matrices is defined only if the number of columns in the first matrix $A$ is equal to the number of rows in the other matrix $B$. If $A$ is $m \times n$ matrix, $B$ is $n \times p$, their product $AB$ is $m \times p$ matrix. $[AB]_{i,j}= A_{i,1} B_{1,j}+   A_{i,2} B_{2,j}+ \ldots + A_{i,n} B_{n,j}= \sum_{r=1}^n  A_{i,r}  B_{r,j}  $，for example $\begin{bmatrix}1 & 0&2 \\-1 & 3&1 \end{bmatrix} \times  \begin{bmatrix}3 & 1 \\2 & 1 \\1&0\end{bmatrix}= \begin{bmatrix}(1 \times 3 & 0 \times 2&2  \times 1)&(1 \times 1 & 0 \times 1&2  \times 0)\\(-1 \times 3 & 3 \times 2&1  \times 1)&(-1 \times 1 & 3 \times 1&1 \times 0) \end{bmatrix}= \begin{bmatrix}5 & 1 \\4 & 2 \end{bmatrix} $

4. exponentiation

That is the same thing as matrix multiplication, for example, $  \begin{bmatrix}1 & 2 \\3 & 4 \end{bmatrix} ^{3} = \begin{bmatrix}1 & 2 \\3 & 4 \end{bmatrix} \begin{bmatrix}1 & 2 \\3 & 4 \end{bmatrix} \begin{bmatrix}1 & 2 \\3 & 4 \end{bmatrix}= \begin{bmatrix}1 \times 1+2 \times 3\ & 1 \times 2+2 \times 4 \\3 \times 1+4 \times 2 & 3  \times 2+4 \times 4\end{bmatrix} \begin{bmatrix}1 & 2 \\3 & 4 \end{bmatrix}  = \begin{bmatrix}7& 10\\15& 22 \end{bmatrix} \begin{bmatrix}1 & 2 \\3 & 4 \end{bmatrix}  = \begin{bmatrix}7 \times 1+10 \times 3 & 7 \times 2+10 \times 4 \\15 \times 1+22 \times 3 & 15 \times 2+22 \times 4 \end{bmatrix} = \begin{bmatrix}37 & 54\\81 & 118 \end{bmatrix}  $

* system of linear equations

A basic application of matrix multiplication is on linear equations, which is a kind of mathematical system of equation. It conforms to the following form: $\begin{cases} a_{1,1} x_{1}+a_{1,2} x_{2}+ \ldots +a_{1,n} x_{n}= b_{1} \\ a_{2,1} x_{1}+a_{2,2} x_{2}+ \ldots +a_{2,n} x_{n}= b_{2} \\ \vdots  \\ a_{m,1} x_{1}+a_{m,2} x_{2}+ \ldots +a_{m,n} x_{n}= b_{m}  \end{cases} $ Where $a_{1,1},a_{1,2}$ and  $b_{1},b_{2}$ and so on is a known constant, while $ x_{1},x_{2}$ and so on is unknown. If expressed by the concepts in linear algebra, the system of linear equations can be written as: $Ax=b$, $A$ is a $m \times n$ matrix, $x$ is a column vector containing $n$ elements, and $b$ is a column vector containing $m$ elements.

$A= \begin{bmatrix} a_{1,1}  & a_{1,2}& \ldots & a_{1,n}\\a_{2,1}  & a_{2,2}& \ldots & a_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\a_{m,1}  & a_{m,2}& \ldots & a_{m,n}\end{bmatrix} $，$x= \begin{bmatrix} x_{1} \\x_{2}\\ \vdots \\x_{n}\end{bmatrix} $，$b= \begin{bmatrix} b_{1}   \\b_{2} \\ \vdots\\ b_{m}  \end{bmatrix} $

It is another way of recording linear equations. Finding the unknown vector $x$ given the matrix $A$, and the vector $b$ is a fundamental problem in linear algebra.


In multiple regression, the method of using a matrix to solve the regression model is described. For samples containing multiple features (explanatory variables), i.e., $n$column, multiple samples(sample size),i.e. row, conventionally expressed as  $y=  \alpha  + \beta _{1} x_{1} + \beta _{2} x_{2}+  \ldots + \beta _{n} x_{n} $，Where $x_{1} ,x_{2}, \ldots ,x_{n}$ is $n$ features, each feature actually contains multiple sample instances ($m$ rows), i.e., $\begin{cases}  Y_{1}= \alpha  + \beta _{1} x_{11} + \beta _{2} x_{12}+  \ldots + \beta _{n} x_{1n} \\Y_{2}=\alpha  + \beta _{1} x_{21} + \beta _{2} x_{22}+  \ldots + \beta _{n} x_{2n}\\ \vdots \\Y_{n}=\alpha  + \beta _{1} x_{m1} + \beta _{2} x_{m2}+  \ldots + \beta _{n} x_{mn} \end{cases} $，Expressed as matrix form is $\begin{bmatrix} Y_{1}   \\Y_{2}\\ \vdots\\Y_{n}  \end{bmatrix} = \begin{bmatrix} \alpha  + \beta _{1} x_{11} + \beta _{2} x_{12}+  \ldots + \beta _{n} x_{1n} \\\alpha  + \beta _{1} x_{21} + \beta _{2} x_{22}+  \ldots + \beta _{n} x_{2n}\\ \vdots \\\alpha  + \beta _{1} x_{m1} + \beta _{2} x_{m2}+  \ldots + \beta _{n} x_{mn}\end{bmatrix} $，It can be simplified even further, $\begin{bmatrix} Y_{1}   \\Y_{2}\\ \vdots\\Y_{n}  \end{bmatrix} =\begin{bmatrix} 1  + x_{11} +x_{12}+  \ldots + x_{1n} \\1  + x_{21} + x_{22}+  \ldots +  x_{2n}\\ \vdots \\1  +  x_{m1} + x_{m2}+  \ldots + x_{mn}\end{bmatrix}  \times  \begin{bmatrix} \alpha  \\  \beta_{1} \\\beta_{2}\\ \vdots \\\beta_{n}  \end{bmatrix} $，It is essentially the multiplication of the matrices.


```python
import sympy
from sympy import Matrix,pprint
A=Matrix([[1,3,1],[1,0,0]])
B=Matrix([[0,0,5],[7,5,0]])
print("Matrix addition:")
pprint(A+B)
print("_"*50)

print("Scalar multiplication of matrix:")
pprint(2*A)
print("_"*50)

C=Matrix([[1,0,2],[-1,3,1]])
D=Matrix([[3,1],[2,1],[1,0]])
print("Multiplication:")
pprint(C*D)
print("_"*50)

E=Matrix([[1,2],[3,4]])
print("Exponentiation:")
pprint(E**3)
print("_"*50)   
```

    Matrix addition:
    ⎡1  3  6⎤
    ⎢       ⎥
    ⎣8  5  0⎦
    __________________________________________________
    Scalar multiplication of matrix:
    ⎡2  6  2⎤
    ⎢       ⎥
    ⎣2  0  0⎦
    __________________________________________________
    Multiplication:
    ⎡5  1⎤
    ⎢    ⎥
    ⎣4  2⎦
    __________________________________________________
    Exponentiation:
    ⎡37  54 ⎤
    ⎢       ⎥
    ⎣81  118⎦
    __________________________________________________
    

#### 1.1.2 Special matrix
1. Zero matrix(null matrix)

A matrix where all the elements are zero, for example,$ \begin{bmatrix}0 & 0 \\0 & 0 \end{bmatrix} $

2. Transpose matrix

Refers to the $m \times n$ matrix A $A= \begin{bmatrix} a_{1,1}  & a_{1,2}& \ldots & a_{1,n}\\a_{2,1}  & a_{2,2}& \ldots & a_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\a_{m,1}  & a_{m,2}& \ldots & a_{m,n}\end{bmatrix} $，By swapping rows and columns $n \times m$，$\begin{bmatrix} a_{1,1}  & a_{2,1}& \ldots &a_{m,1} \\ a_{1,2}  & a_{2,2}& \ldots &a_{m,2} \\ \vdots & \vdots & \ddots & \vdots \\ a_{1,n}  & a_{2,n}& \ldots &a_{m,n}\end{bmatrix} $，You can use $A^{T} $ to express, for example, the transpose matrix of $3 \times 2$ matrix $\begin{bmatrix}1 & 2 \\3 & 4\\5&6 \end{bmatrix} $ is $2 \times 3$ matrix B $\begin{bmatrix}1& 3&5 \\2 & 4&6 \end{bmatrix} $。

3. Symmetric matrix

A symmetric square $n$ order matrix with diagonal elements as the centerline, such as $\begin{bmatrix}1 & 5&6&7 \\5 & 2&8&9\\6&8&3&10\\7&9&10&4 \end{bmatrix} $, the symmetric matrix is identical to its transpose matrix.

4. Ttriangular matrix

For example,$\begin{bmatrix}1 & 5&6&7 \\0 & 2&8&9\\0&0&3&10\\0&0&0&4 \end{bmatrix} $，A $n$ order matrix in which all the elements in the lower-left corner of a diagonal element are 0.

For example,$\begin{bmatrix}1 & 0&0&0 \\5 & 2&0&0\\6&8&3&0\\7&9&10&4 \end{bmatrix}$,A $n$ order matrix in which all the elements in the upper-right corner of a diagonal element are 0.

6. Diagonal matrix

For example, $\begin{bmatrix}1 & 0&0&0 \\0& 2&0&0\\0&0&3&0\\0&0&0&4 \end{bmatrix}$，A $n$ order matrix in which all elements other than diagonal elements are 0 can be expressed as $diag(1,2,3,4)$. The $p$ power of the diagonal matrix is equal to the $p$ power of the diagonal elements, and the formula is: $\begin{bmatrix} a_{1,1}  & 0&0&0 \\0& a_{2,2}&0&0\\0&0&a_{3,3}&0\\0&0&0&a_{n,n} \end{bmatrix}^{p} =\begin{bmatrix} a_{1,1}^p  & 0&0&0 \\0& a_{2,2}^p&0&0\\0&0&a_{3,3}^p&0\\0&0&0&a_{n,n}^p \end{bmatrix}$，such as, $\begin{bmatrix}2 & 0 \\0 & 3 \end{bmatrix} ^{2} = \begin{bmatrix}2^{2}  &  0  \\0&  3^{2}  \end{bmatrix} = \begin{bmatrix}4 & 0 \\0& 9 \end{bmatrix} $

7. Identity matrix

For example, $\begin{bmatrix}1 & 0&0&0 \\0 & 1&0&0 \\ 0&0&1&0\\0&0&0&1   \end{bmatrix} $，A $n$ order square matrix in which the diagonal elements are all 1, and all other elements other than diagonal elements are 0 is an identity matrix, namely $diag(1,1 ,\ldots ,1)$.  The identity matrix times any matrix; it does not make any difference to the matrix.

8. Inverse matrix

Also known as the inverse matrix, in linear algebra, given a $n$ order square matrix $A$, if there is a $n$ order square matrix B, so that $AB=BA=I_n$, where $I_n$ is $n$ order identity matrix, then $A$ is invertible, and $B$ is the inverse matrix of A, denoted as $A^{-1}$. Only $n \times n $ square matrix can have an inverse matrix. If the inverse matrix of a square matrix $A$ exists, then $A$ is said to be a non-singular square matrix or invertible matrix. The inverse matrix can be solved by the cofactor method(unpractical), elimination method, etc. If you are interested in the solution, see *The Manga Guide to Linear Algebra*. In the code world, directly use the method provided by Sympy.


Not all square matrices have an inverse matrix, where we can use determinant,  abbreviated as det, and use the det() method provided by Sympy library to calculate and judge.


```python
print("Zero matrix:")
pprint(sympy.zeros(2))

print("Transpose matrix:")
F=Matrix([[1,2],[3,4],[5,6]])
pprint(F.T)

print("Transpose matrix of a symmetric matrix:")
G=Matrix([[1,5,6,7],[5,2,8,9],[6,8,3,10],[7,9,10,4]])
pprint(G.T)

print("A diagonal matrix in the power of 2:")
H=Matrix([[2,0],[0,3]])
pprint(H**2)

print("Identity matrix:")
pprint(sympy.eye(4))

print("The identity matrix multiplied by any matrix does not affect the matrix:")
pprint(sympy.eye(4)*G)

print("Solve the inverse matrix and multiply it by itself:")
print("Calculate the inverse matrix using the -1 power:")
G_inverse=G**-1
pprint(G*G_inverse)
print(".inv() is used to calculate the inverse matrix:")
G_inverse_=G.inv()
pprint(G*G_inverse_)

print("Judge whether the square matrix has an inverse matrix:")
print(G.det())
print(G_inverse.det())
```

    Zero matrix:
    ⎡0  0⎤
    ⎢    ⎥
    ⎣0  0⎦
    Transpose matrix:
    ⎡1  3  5⎤
    ⎢       ⎥
    ⎣2  4  6⎦
    Transpose matrix of a symmetric matrix:
    ⎡1  5  6   7 ⎤
    ⎢            ⎥
    ⎢5  2  8   9 ⎥
    ⎢            ⎥
    ⎢6  8  3   10⎥
    ⎢            ⎥
    ⎣7  9  10  4 ⎦
    A diagonal matrix in the power of 2:
    ⎡4  0⎤
    ⎢    ⎥
    ⎣0  9⎦
    Identity matrix:
    ⎡1  0  0  0⎤
    ⎢          ⎥
    ⎢0  1  0  0⎥
    ⎢          ⎥
    ⎢0  0  1  0⎥
    ⎢          ⎥
    ⎣0  0  0  1⎦
    The identity matrix multiplied by any matrix does not affect the matrix:
    ⎡1  5  6   7 ⎤
    ⎢            ⎥
    ⎢5  2  8   9 ⎥
    ⎢            ⎥
    ⎢6  8  3   10⎥
    ⎢            ⎥
    ⎣7  9  10  4 ⎦
    Solve the inverse matrix and multiply it by itself:
    .inv() is used to calculate the inverse matrix:
    ⎡1  0  0  0⎤
    ⎢          ⎥
    ⎢0  1  0  0⎥
    ⎢          ⎥
    ⎢0  0  1  0⎥
    ⎢          ⎥
    ⎣0  0  0  1⎦
    使用.inv()计算逆矩阵:
    ⎡1  0  0  0⎤
    ⎢          ⎥
    ⎢0  1  0  0⎥
    ⎢          ⎥
    ⎢0  0  1  0⎥
    ⎢          ⎥
    ⎣0  0  0  1⎦
    Judge whether the square matrix has an inverse matrix:
    -3123
    -1/3123
    

### 1.2 Euclidean vector
#### 1.2.1 Vector concepts, expressions, and operations
Generally, at the same time, the geometrical objects satisfy with characteristics of both magnitude and direction can be regarded as a vector, it said after specifies a coordinate system, with the coordinates of a vector in the coordinates system to represent the vector, both abstract and geometric figuration, have the highest practical, widely used in the quantitative analysis. For a free vector, the vector can be represented by a point in a coordinate system whose coordinate value is the endpoint coordinate of the vector when the vector's starting point is shifted to the origin of coordinates.


Assume a vector $\vec{a}$ with a coordinate system $S$. Defined several special basic vectors (referred to as the base vector which together constitutes the base in this coordinate system) $\vec{ e_{1} },\vec{ e_{2} }, \ldots ,\vec{ e_{n} }$, the projection of the vector under different base vector is the corresponding coordinate system. Each projection value constitutes an ordered array(coordinate) that can be uniquely represented by the vector under the coordinate system $S$, and corresponds to its endpoint. In other words, other vectors can be represented by simply stretching these vectors and then adding them following the parallelogram rule (commonly referred to as "representing a vector with basal linear", i.e. the vector is some linear combination of the base vectors), namely, $\vec{ a}=a_{1} \vec{ e_{1} },a_{2} \vec{ e_{2} }, \ldots ,a_{n} \vec{ e_{n} }$，Where $\vec{ a_{1} },\vec{ a_{2} }, \ldots ,\vec{ a_{n} }$ is the corresponding projection of $\vec{ a}$ in the $\vec{ e_{1} },\vec{ e_{2} } \ldots ,\vec{ e_{n} }$ respectively. When the base is known, the symbol of each base vector can be directly omitted, similar to the point in the coordinate system, and directly expressed as $\vec{a}=( a_{1}, a_{2}, \ldots , a_{n} )$ with coordinate. In matrix operations, vectors are more often written as a matrix-like column or row vectors.  A vector referred to linear algebra usually defaults to a column vector. For example, a vector $\vec{a}=(a,b,c)$ can be written as $\vec{a}= \begin{bmatrix}a \\b\\c \end{bmatrix} $，$\vec{a}= \begin{bmatrix}a &b&c \end{bmatrix} $, where the first is a column vector style, and the second is a row vector style. $n$ dimension column vector can be regarded as $n \times 1$ matrix, $n$ dimension row vector can be regarded as $1\times n$ matrix.

In the common three-dimensional rectangular system $\bigcirc xyz$(three-dimensional Cartesian coordinate system), the basic vector is the unit vector $\vec{i},\vec{j},\vec{k}$ of length 1 with the abscissa axis($\bigcirc x$), the ordinate axis($\bigcirc y$), and the applicate axis($\bigcirc z$) in the direction, namely the base vector. Once these three vectors are taken, the rest of the vectors can be represented through an array of triples, because they can be represented as the sum of certain multiples of the three fundamental vectors. Such as a labeled vector (2,1,3) is 2 vectors $\vec{i}$ plus 1 vector $\vec{j}$ plus 3 vectors $\vec{k}$, namely $(a,b,c)=a\vec{i}+b\vec{j}+c\vec{k}$. So the vector is essentially a matrix, and the calculation of the vector is the same as the calculation of the matrix.

There are two ways provided by the Sympy library for vector calculation. One is to use matrices entirely, since they are the same calculations, such as sum, difference, multiples, and products; The other is that the library provides a special class 'Vector' for vectors, which can realize the expression of vectors and basic operations on vectors through the methods provided by 'Vector'. A vector's main purpose is to solve space geometry, so understanding vector should be combined with space geometry and expression to understand the cause and effect effectively. Simply looking at vector expressions is not a very rational way to learn. For this part, after understanding the basic concepts of vectors, coordinate spaces, and operations, if you want to express and calculate vectors in code, it is necessary to read Sympy corresponding ['Vector'](https://docs.sympy.org/dev/modules/vector/index.html) and ['Matrices' (linear algebra)](https://docs.sympy.org/latest/modules/matrices/matrices.html) two parts, then read the following code will be relatively easy.


The `vector_plot_3d` function is defined to facilitate the use of Matplotlib to print vectors in 3D space. Firstly, the 3d coordinate system named 'C' is defined, and the base vector 'i,j,k' are obtained, and the base vectors are used to construct the vector 'v1'. At the same time, by extracting this vector's coefficient in the three axes, the components of the v1 vector on each axis (projection vector) are established, and the sum of projection vectors of each axis is the v1 vector. Print its vector graph in three-dimensional space, observe the relationship between vectors. Note that in determining the vector space's starting position, vector_k starts with the vector of the sum of vector_i and vector_j, and vector_j starts with vector_i.

* All components collection of the $n \times 1$ vector $\begin{bmatrix} a_{1} \\ a_{2} \\ \vdots \\ a_{n} \end{bmatrix} $ represented as $ \\R^{n} $.



```python
import matplotlib.pyplot as plt
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Vector, BaseVector
from sympy.vector import Vector

fig, ax=plt.subplots(figsize=(12,12))
ax=fig.add_subplot( projection='3d')

def vector_plot_3d(ax_3d,C,origin_vector,vector,color='r',label='vector',arrow_length_ratio=0.1):
    '''
    funciton - Transform vector and Matrix data formats in Sympy to Matplotlib printable  data formats
    
    Paras:
    ax_3d - Matplotlib 3d subgraph
    C - /coordinate_system - Coordinate systems defined under the Sympy
    origin_vector - If it is a fixed vector, the vector's starting point is given (using the vector, which indicates the position from the origin of coordinates). If it is a free vector, the starting point is set to the origin of coordinates.
    vector - The vector to be printed
    color - Vector color
    label - Vector label
    arrow_length_ratio - Vector arrow size
    '''
    origin_vector_matrix=origin_vector.to_matrix(C)
    x=origin_vector_matrix.row(0)[0]
    y=origin_vector_matrix.row(1)[0]
    z=origin_vector_matrix.row(2)[0]
    
    vector_matrix=vector.to_matrix(C)
    u=vector_matrix.row(0)[0]
    v=vector_matrix.row(1)[0]
    w=vector_matrix.row(2)[0]
    ax_3d.quiver(x,y,z,u,v,w,color=color,label=label,arrow_length_ratio=arrow_length_ratio)


#Define the coordinate system and print the vector v1=3*i+4*j+5*k
C=CoordSys3D('C')
i, j, k = C.base_vectors()
v1=3*i+4*j+5*k
v1_origin=Vector.zero
vector_plot_3d(ax,C,v1_origin,v1,color='r',label='vector',arrow_length_ratio=0.1)

#Print the projection of vector V1=3*i+4*j+5*k onto the axis
v1_i=v1.coeff(i)*i
vector_plot_3d(ax,C,v1_origin,v1_i,color='b',label='vector_i',arrow_length_ratio=0.1)

v1_j=v1.coeff(j)*j
vector_plot_3d(ax,C,v1_i,v1_j,color='g',label='vector_j',arrow_length_ratio=0.1)

v1_k=v1.coeff(k)*k
vector_plot_3d(ax,C,v1_i+v1_j,v1_k,color='yellow',label='vector_k',arrow_length_ratio=0.1)

ax.set_xlim3d(0,5)
ax.set_ylim3d(0,5)
ax.set_zlim3d(0,5)

ax.legend()
ax.view_init(20,20) #The angle of the graph can be rotated for easy observation
plt.show()
```


<a href=""><img src="./imgs/9_1.png" height="auto" width="auto" title="caDesign"></a>


#### 1.2.2 Linear independence
In linear algebra, a set of elements in the vector space, without a vector by a finite number of a linear combination of the other vector, can be used, says it is called linear independence(linearly independent), whereas called linear dependence(linearly dependent). The assumption on $V$ in the vector space of domain $K$, if the domain $K$ has not all zero elements $a_{1} ,a_{2} , \ldots ,a_{n} $ , makes $a_{1}v_{1}+a_{2} v_{2}+ \ldots +a_{n}v_{n}=0 $, or said, $\sum_{i=1}^n a_{i} v_{i} =0$, where $v_{1} ,v_{2} , \ldots ,v_{n} $ is the vector of $V$, we will call them linear independence. The 0 on the right is $\vec 0$, namely a vector of 0, instead of 0 scalar. If there is no such elements of $K$, then $v_{1} ,v_{2} , \ldots ,v_{n} $ are linear indenpendence. Linear independence can be defiend more directly. Vector $v_{1} ,v_{2} , \ldots ,v_{n} $ are linear independence, if and only if they meet the conditions: if $a_{1} ,a_{2} , \ldots ,a_{n} $ are elements of $K$, suitable for $a_{1}v_{1}+a_{2} v_{2}+ \ldots +a_{n}v_{n}=0 $, so for all $i=1,2, \ldots ,n$ have $a_{i} =0$.

For an infinite set of $V$, if any of its finite subsets are linear independence, then the original infinite set is linearly independent. Linear dependence is an important concept in linear algebra because a linearly independent vector set can generate a vector space. This set of vectors is the base of the vector space.

First, a 3d coordinate system was established through the 'CoordSys3D' method in Sympy(in which unit vectors $\vec{i},\vec{j},\vec{k}$ can be extracted directly); based on this coordinate system, the vector set ($V$), including v2,v3,v4,v5, is established by multiples of unit vectors. If the multiple is equal to 0, for example, `v2=a*1*N.i+a*0*N.j`, 0 is also maintained to maintain consistency in all directions, making it easy to observe. By printing individual vectors, we can view the expression form of variables in Sympy. The vector collection is linearly dependent, which can be given in addition to $a=b=c=d=0$, the other a,b,c,d value, that is not all zero elements, such as a_,b_,c_,d_=1,2,0,-1, or a_,b_,c_,d_=1,-3,-1,2, also meet $a_{1}v_{1}+a_{2} v_{2}+ \ldots +a_{n}v_{n}=0 $, the above a,b,c,d is the $a_{1} ,a_{2} , \ldots ,a_{n} $. You can therefore print the graphs by defining the `move_alongVectors` function implementation, where you will see the first and second graphs forming a closed 2d graph(back to the starting point). Also in the 3d space, the vector set v6,v7,v8,v9 are defined. Because of linear dependence, such as the third figure, a closed polyline is formed(back to the starting point).
 

To determine whether a vector set $V$, is linearly independent, you can use `matrix.rref()` to transform the matrix representing vectors into a Row Echelon Form. If there are all zero values at the end of the returned reduced Row Echelon Form, the vector set is linearly dependent. For example, using `matrix.to_matrix(C)` method to convert v2,v3,v4,v5 and other vectors into matrix expression, and using `matrix.col_insert()` method to merge them int matrix, using `matrix.rref()` to calculate, for the dataset v2,v3,v4,v5, the result of its reduced Row Echelon Form is $\begin{bmatrix}1 & 0&0 \\0 & 1&0\\0&0&1\\0&0&0 \end{bmatrix} $, the end contains all zero rows, therefore, the dataset is linearly dependent. The list (0,1,2) of pivot locations is also returned. However, the vector set composed of v10,v11, does not contain all-zero rows at the end, so it is linearly independent. It is impossible to construct a closed spacial polyline through each vector so that the vector space can be generated, and the vector set is the base of this vector space. For any element vector in $\\R_{m} $ $\begin{bmatrix} y_{1} \\ y_{2}\\ \vdots \\ y_{m}   \end{bmatrix}$，When the solutions $c_{1} ,c_{2}, \ldots ,c_{n}  $ of  $\begin{bmatrix} y_{1} \\ y_{2}\\ \vdots \\ y_{m}   \end{bmatrix} = c_{1} \begin{bmatrix} a_{11} \\ a_{21}\\ \vdots \\ a_{m1}   \end{bmatrix} + c_{2} \begin{bmatrix} a_{12} \\ a_{22}\\ \vdots \\ a_{m2}   \end{bmatrix} + \ldots + c_{n} \begin{bmatrix} a_{1n} \\ a_{2n}\\ \vdots \\ a_{mn}   \end{bmatrix} $ are all zero，the set  $\Bigg\{ \begin{bmatrix} a_{11} \\ a_{21}\\ \vdots \\ a_{m1}   \end{bmatrix},\begin{bmatrix} a_{12} \\ a_{22}\\ \vdots \\ a_{m2}   \end{bmatrix} , \ldots , \begin{bmatrix} a_{1n} \\ a_{2n}\\ \vdots \\ a_{mn}   \end{bmatrix} \Bigg\}$ is called the base，that is, to represent the set in $\\R_{m} $ composed by the minimum vectors required to any element.

* Echelon Form matrix

In linear algebra, a matrix is called Row Echelon Form if it meets the following conditions:

1. All non-zero rows(up to at least one non-zero element of the matrix) are on top of all zero rows. So all the zero rows are at the bottom of the matrix.
2. The leading coefficient of non-zero rows, also known as the pivot element, is the first non-zero element on the far left, which is strictly further to the right than the first coefficient on the above row(some versions require that the first coefficient of the non-zero row must be 1).
3. In the column where the first coefficient is, the elements below the first coefficient are all zero(corollary to the first two).

For example: $\begin{bmatrix}1 &  a_{1} &a_{2} | & b_{1}   \\0 &  2 &a_{3} | & b_{2} \\0 &  0 &1| & b_{3}  \end{bmatrix} $。

> The augmented matrix, also known as extended matrix, is the matrix obtained by filling the right side of the coefficient matrix in linear algebra with the constant sequence on the right side of the equal sign of linear equations. For example, the coefficient matrix of the equation $AX=B$ is $A$, and its augmented matrix is $A | B$. The equation set uniquely determines the augmented matrix. The augmented matrix's elementary row transformation can be used to judge whether the corresponding linear system of equations has a solution and simplifies the original equation's solution. Its vertical line can be omitted.

The reduced row echelon form is also known as row canonical form, if an additional condition is met: each leading item coefficient is 1 and is the only non-zero element of its column, for example, $\begin{bmatrix}1& 0& a_{1} &0& | & b_{1} \\0& 1&0&0& | &b_{2} \\0&0&0&1& | & b_{3} \end{bmatrix} $, note that the left part of the reduced row echelon form (coefficient part) does not mean that it is always the identity matrix.

Any matrix can be transformed into row echelon form by finite-step row elementary transformations. Due to the elementary transformation retain the line space of the matrix, the line space of the row echelon matrix is the same as the original matrix before the transformation(this is also using 'sympy.rref()' to calculate reduced row echelon form), need to transpose the vector set with shape $\begin{bmatrix} a_{1}i & a_{2}i & a_{3}i \\ a_{1}j & a_{2}j & a_{3}j\\ \vdots & \vdots & \vdots \\ a_{1}k & a_{2}k& a_{3}k \end{bmatrix} $ into $\begin{bmatrix} a_{1}i & a_{2}i & a_{3}i \\ a_{1}j & a_{2}j & a_{3}j\\ \vdots & \vdots & \vdots \\ a_{1}k & a_{2}k& a_{3}k \end{bmatrix} $, keep matrix row as $ a_{n} \vec{i}, a_{n}\vec{j}, a_{n}\vec{k}$）. Row echelon form is not unique. For example, row echelon form multiplied by a scalar coefficient is still a row echelon form. But, it turns out that the reduced row echelon form of a matrix is unique.



```python
from sympy.vector import CoordSys3D
from sympy.abc import a, b, c,d,e,f,g,h,o,p #The equivalent of a,b=sympy.symbols(["a","b"])
from sympy.vector import Vector
from sympy import pprint,Eq,solve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig, axs=plt.subplots(1,3,figsize=(30,10))
axs[0]=fig.add_subplot(1,3,1, projection='3d')
axs[1]=fig.add_subplot(1,3,2, projection='3d')
axs[2]=fig.add_subplot(1,3,3, projection='3d')

#A - In 2d, draw the solutions of v2+v3+v4+v5=0, linearly dependent, with multiple solutions.
N=CoordSys3D('N')
v2=a*1*N.i+a*0*N.j
v3=b*0*N.i+b*1*N.j
v4=c*3*N.i+c*1*N.j
v5=d*1*N.i+d*2*N.j
v0=Vector.zero

def move_alongVectors(vector_list,coeffi_list,C,ax,):
    import random
    import sympy
    '''
    function - Given the vector and the corresponding coefficient, draw along the vector.
    
    Paras:
    vector_list - List of vectors, in moving order.
    coeffi_list - Coefficient of vector   
    C - The coordinate system defined under Sympy
    ax - Subgraph
    '''
    colors=[color[0] for color in mcolors.TABLEAU_COLORS.items()]  #mcolors.BASE_COLORS, mcolors.TABLEAU_COLORS,mcolors.CSS4_COLORS
    colors__random_selection=random.sample(colors,len(vector_list)-1)
    v_accumulation=[]
    v_accumulation.append(vector_list[0])
    #Each vector is drawn, starting with the sum of all the previous vectors.
    for expr in vector_list[1:]:
        v_accumulation.append(expr+v_accumulation[-1])
    
    v_accumulation=v_accumulation[:-1]   
    for i in range(1,len(vector_list)):
        vector_plot_3d(ax,C,v_accumulation[i-1].subs(coeffi_list),vector_list[i].subs(coeffi_list),color=colors__random_selection[i-1],label='v_%s'%coeffi_list[i-1][0],arrow_length_ratio=0.2)
        
#v2+v3+v4+v5=0，The solution-1 when the sum of the vectors is 0
vector_list=[v0,v2,v3,v4,v5]
a_,b_,c_,d_=1,2,0,-1        
coeffi_list=[(a,a_),(b,b_),(c,c_),(d,d_)]
move_alongVectors(vector_list,coeffi_list,N,axs[0],)    
    
#v2+v3+v4+v5=0，The solution-2 when the sum of the vectors is 0
vector_list=[v0,v2,v3,v4,v5]
a_,b_,c_,d_=1,-3,-1,2      
coeffi_list=[(a,a_),(b,b_),(c,c_),(d,d_)]
move_alongVectors(vector_list,coeffi_list,N,axs[1],)       

#B - 3D, draw the solution of v6+v7+v8+v9=0, linear dependence
M=CoordSys3D('M')
v6=e*1*M.i+e*0*M.j+e*0*M.k
v7=f*0*M.i+f*1*M.j+f*0*M.k
v8=g*3*M.i+g*1*M.j+g*-3*M.k
v9=h*0*M.i+h*0*M.k+h*3*M.k
v0=Vector.zero

vector_list=[v0,v6,v7,v8,v9]
e_,f_,g_,h_=3,1,-1,-1
coeffi_list=[(e,e_),(f,f_),(g,g_),(h,h_)]
move_alongVectors(vector_list,coeffi_list,M,axs[2],)  
      
#C - The vector is converted to a matrix, and judge whether it is linearly independent.
#C-1 - For the vector set v2,v3,v4,v5
v_2345_matrix=v2.to_matrix(N)
for v in [v3,v4,v5]:
    v_temp=v.to_matrix(N)
    v_2345_matrix=v_2345_matrix.col_insert(-1,v_temp)
print("v_2345_matrix:")
pprint(v_2345_matrix)
print("v_2345_matrix.T rref:")
pprint(v_2345_matrix.T.rref())

#C-2 - For the vector setv6,v7,v8,v9
print("_"*50)
v_6789_matrix=v6.to_matrix(M)
for v in [v7,v8,v9]:
    v_temp=v.to_matrix(M)
    v_6789_matrix=v_6789_matrix.col_insert(-1,v_temp)
print("v_6789_matrix:")
pprint(v_6789_matrix)
print("v_6789_matrix.T rref:")
pprint(v_6789_matrix.T.rref())

#C-3 For the vector set v10,v11
print("_"*50)
C=CoordSys3D('C')
v10=o*1*C.i+o*0*C.j
v11=p*0*C.i+p*1*C.j
v_10_11_matrix=v10.to_matrix(C).col_insert(-1,v11.to_matrix(C))
print("v_10_11_matrix:")
pprint(v_10_11_matrix)
print("v_10_11_matrix.T rref:")
pprint(v_10_11_matrix.T.rref())

axs[0].set_xlim3d(0,2)
axs[0].set_ylim3d(0,2)
axs[0].set_zlim3d(0,5)

axs[1].set_xlim3d(-3,1)
axs[1].set_ylim3d(-4,1)
axs[1].set_zlim3d(0,5)

axs[2].set_xlim3d(0,4)
axs[2].set_ylim3d(0,4)
axs[2].set_zlim3d(0,4)

axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[0].view_init(90,0) #The angle of the graph can be rotated for easy observation
axs[1].view_init(90,0)
axs[2].view_init(30,50)
plt.show()
```

    v_2345_matrix:
    ⎡0  3⋅c   d   a⎤
    ⎢              ⎥
    ⎢b   c   2⋅d  0⎥
    ⎢              ⎥
    ⎣0   0    0   0⎦
    v_2345_matrix.T rref:
    ⎛⎡1  0  0⎤        ⎞
    ⎜⎢       ⎥        ⎟
    ⎜⎢0  1  0⎥        ⎟
    ⎜⎢       ⎥, (0, 1)⎟
    ⎜⎢0  0  0⎥        ⎟
    ⎜⎢       ⎥        ⎟
    ⎝⎣0  0  0⎦        ⎠
    __________________________________________________
    v_6789_matrix:
    ⎡0  3⋅g    0   e⎤
    ⎢               ⎥
    ⎢f   g     0   0⎥
    ⎢               ⎥
    ⎣0  -3⋅g  3⋅h  0⎦
    v_6789_matrix.T rref:
    ⎛⎡1  0  0⎤           ⎞
    ⎜⎢       ⎥           ⎟
    ⎜⎢0  1  0⎥           ⎟
    ⎜⎢       ⎥, (0, 1, 2)⎟
    ⎜⎢0  0  1⎥           ⎟
    ⎜⎢       ⎥           ⎟
    ⎝⎣0  0  0⎦           ⎠
    __________________________________________________
    v_10_11_matrix:
    ⎡0  o⎤
    ⎢    ⎥
    ⎢p  0⎥
    ⎢    ⎥
    ⎣0  0⎦
    v_10_11_matrix.T rref:
    ⎛⎡1  0  0⎤        ⎞
    ⎜⎢       ⎥, (0, 1)⎟
    ⎝⎣0  1  0⎦        ⎠
    


<a href=""><img src="./imgs/9_2.png" height="auto" width="auto" title="caDesign"></a>


The above code establishes a vector dataset using 'sympy.vector' method, which enables intuitive understanding of vector space and vector operations in space. The following code directly establishes the vector coefficient matrix of the matrix mode. It conducts the relevant calculation, including obtaining the row echelon form matrix, reduced echelon form matrix, and solving the coefficient, so that it can be judged that the vector set V_C_matrix is linearly dependent and has a non-complete zero solution. The graph returning to the starting point can be constructed in the space.


```python
C=CoordSys3D('C')
V_C_matrix=Matrix([[1,4,2,-3],[7,10,-4,-1],[-2,1,5,-4]])

lambda_1,lambda_2,lambda_3=sympy.symbols(['lambda_1','lambda_2','lambda_3'])
coeffi_expr=V_C_matrix.T*Matrix([lambda_1,lambda_2,lambda_3])
print("Vector set(linear equations) coefficient matrix:")
pprint(coeffi_expr)
print("echelon_form:")
pprint(coeffi_expr.echelon_form())
print("reduced row echelon form:")
pprint(coeffi_expr.rref())

#Solving systems of linear equations
print("For solutions of linear equations(matrix patterns), Sympy provide several ways:")
pprint(solve(coeffi_expr,(lambda_1,lambda_2,lambda_3)))
pprint(solve(coeffi_expr,set=True))
from sympy import Matrix, solve_linear_system
pprint(solve_linear_system(V_C_matrix.T,lambda_1,lambda_2,lambda_3))
```

    Vector set(linear equations) coefficient matrix:
    ⎡ λ₁ + 7⋅λ₂ - 2⋅λ₃ ⎤
    ⎢                  ⎥
    ⎢4⋅λ₁ + 10⋅λ₂ + λ₃ ⎥
    ⎢                  ⎥
    ⎢2⋅λ₁ - 4⋅λ₂ + 5⋅λ₃⎥
    ⎢                  ⎥
    ⎣-3⋅λ₁ - λ₂ - 4⋅λ₃ ⎦
    echelon_form:
    ⎡λ₁ + 7⋅λ₂ - 2⋅λ₃⎤
    ⎢                ⎥
    ⎢       0        ⎥
    ⎢                ⎥
    ⎢       0        ⎥
    ⎢                ⎥
    ⎣       0        ⎦
    reduced row echelon form:
    ⎛⎡1⎤      ⎞
    ⎜⎢ ⎥      ⎟
    ⎜⎢0⎥      ⎟
    ⎜⎢ ⎥, (0,)⎟
    ⎜⎢0⎥      ⎟
    ⎜⎢ ⎥      ⎟
    ⎝⎣0⎦      ⎠
    For solutions of linear equations(matrix patterns), Sympy provide several ways:
    ⎧    -3⋅λ₃       λ₃⎫
    ⎨λ₁: ──────, λ₂: ──⎬
    ⎩      2         2 ⎭
    ⎛          ⎧⎛-3⋅λ₃   λ₃⎞⎫⎞
    ⎜[λ₁, λ₂], ⎨⎜──────, ──⎟⎬⎟
    ⎝          ⎩⎝  2     2 ⎠⎭⎠
    {λ₁: 3/2, λ₂: -1/2}
    

#### 1.2.3 Dimension
* Subspaces(linear subspaces, or vector subspaces)

Assuming $c$ is any real number. If the subset $W$ of $\\R_{m} $ satisfies the following three conditions: 1. $c$ times any element of  $W$ is also an element of $W$; 2. The sum of any element of $W$ is also an element of $W$; 3. Zero vector $\vec 0$ is in $W$. That satisfy the three conditions, 1, if $ \begin{bmatrix} a_{1i} \\a_{2i}\\ \vdots \\a_{mi} \end{bmatrix} \in W $, then $c\begin{bmatrix} a_{1i} \\a_{2i}\\ \vdots \\a_{mi} \end{bmatrix} \in W$; 2. if $ \begin{bmatrix} a_{1i} \\a_{2i}\\ \vdots \\a_{mi} \end{bmatrix} \in W $ and $ \begin{bmatrix} a_{1j} \\a_{2j}\\ \vdots \\a_{mj} \end{bmatrix} \in W$, then $ \begin{bmatrix} a_{1i} \\a_{2i}\\ \vdots \\a_{mi} \end{bmatrix} +\begin{bmatrix} a_{1j} \\a_{2j}\\ \vdots \\a_{mj} \end{bmatrix} \in W$; At the same time  $\vec0 \in W$, then $W$ is called $\\R_{m} $ linear subspaces, abbreviated as subspace. We can understand the subspace from 2 or 3 dimensional space first, if it multiply by a multiple, which is really just a scalling of the vector; as for the sum of vectors, they only travel along multiple vectors, and these calculation exist in one spatial dimension, which can help us understand the expansion to more than 3 dimensions(which can not be directly observed like 2 and 3 dimensions), such as "the line through the origin", "the plane through the origin" and so on.

$W$ linearly independent element, namely base element, $\Bigg\{ \begin{bmatrix} a_{11} \\ a_{21}\\ \vdots \\ a_{m1}   \end{bmatrix},\begin{bmatrix} a_{12} \\ a_{22}\\ \vdots \\ a_{m2}   \end{bmatrix} , \ldots , \begin{bmatrix} a_{1n} \\ a_{2n}\\ \vdots \\ a_{mn}   \end{bmatrix} \Bigg\}$，The number $n$ of base elements is called the subspace $ W $ dimension and is typically expressed as $dimW$(dim is the dimension abbreviation).

Such as $W$ is the $\\R^{3} $ subspace, the vector vector_a=0*C.i+3*C.j+1*C.k, namely $\begin{bmatrix}0 \\3\\1 \end{bmatrix} $ and the vector vector_b=0*C.i+1*C.j+2*C.k, namely $\begin{bmatrix}0 \\1\\2 \end{bmatrix} $ is linearly independent(viewed throug `pprint(v_a_b.T.rref())`, without all zero lines) element. Obviously, $\Bigg\{ c_{1} \begin{bmatrix}0 \\3\\1 \end{bmatrix}+ c_{2} \begin{bmatrix}0 \\1\\2 \end{bmatrix} c_{1}, c_{2} for any real number \Bigg\}$ equation is satisfied, so the set $\Bigg\{ \begin{bmatrix}0 \\3\\1 \end{bmatrix}, \begin{bmatrix}0 \\1\\2 \end{bmatrix}  \Bigg\}$ is the base for the subspace $W$ with a dimension of 2.


```python
from matplotlib.patches import PathPatch,Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
fig, ax=plt.subplots(figsize=(12,12))
ax=fig.add_subplot( projection='3d')
p=Rectangle((-5, -5), 10,10,color='red', alpha=0.1)
ax.add_patch(p)
art3d.pathpatch_2d_to_3d(p, z=0, zdir="x",)

C=CoordSys3D('C')
vector_0=Vector.zero
vector_1=5*C.j
vector_plot_3d(ax,C,vector_0,vector_1,color='gray',label='vector_j',arrow_length_ratio=0.1)
vector_2=5*C.k
vector_plot_3d(ax,C,vector_0,vector_2,color='gray',label='vector_k',arrow_length_ratio=0.1)

vector_a=0*C.i+3*C.j+1*C.k
vector_b=0*C.i+1*C.j+2*C.k
vector_plot_3d(ax,C,vector_0,vector_a,color='green',label='vector_a',arrow_length_ratio=0.1)
vector_plot_3d(ax,C,vector_0,vector_b,color='olive',label='vector_b',arrow_length_ratio=0.1)

ax.set_xlim3d(-5,5)
ax.set_ylim3d(-5,5)
ax.set_zlim3d(-5,5)

ax.legend()
ax.view_init(20,20) #The angle of the graph can be rotated for easy observation
plt.show()

v_a_b=vector_a.to_matrix(C).col_insert(-1,vector_b.to_matrix(C))
print("v_a_b.T rref:")
pprint(v_a_b.T.rref())
```


<a href=""><img src="./imgs/9_3.png" height="auto" width="auto" title="caDesign"></a>


    v_a_b.T rref:
    ⎛⎡0  1  0⎤        ⎞
    ⎜⎢       ⎥, (1, 2)⎟
    ⎝⎣0  0  1⎦        ⎠
    

### 1.3 Linear mapping
#### 1.3.1 Definition, the matrix calculation method of linear mapping, image
Assuming $\begin{bmatrix} x_{1i} \\x_{2i}\\ \vdots \\x_{ni} \end{bmatrix} $和$\begin{bmatrix} x_{1j} \\x_{2j}\\ \vdots \\x_{nj} \end{bmatrix} $ is any element of $\\R^{n} $, $f$ is the mapping from $\\R^{n} $ to $\\R^{m} $. When the mapping $f$ meets the following two conditions, says the mapping $f$ is a linear mapping from $\\R^{n} $ to $\\R^{m} $. 1. $f \Bigg\{ \begin{bmatrix}x_{1i} \\x_{2i}\\ \vdots \\x_{ni} \end{bmatrix} \Bigg\}+f \Bigg\{ \begin{bmatrix}x_{1j} \\x_{2j}\\ \vdots \\x_{nj} \end{bmatrix} \Bigg\}$ and $f \Bigg\{  \begin{bmatrix}x_{1i} +x_{1j} \\x_{2i}+x_{2j}\\ \vdots \\x_{ni}+x_{nj} \end{bmatrix} \Bigg\} $ is equal; 2. $cf \Bigg\{ \begin{bmatrix}x_{1i} \\x_{2i}\\ \vdots \\x_{ni} \end{bmatrix} \Bigg\} $ and $f \Bigg\{ c \begin{bmatrix}x_{1i} \\x_{2i}\\ \vdots \\x_{ni} \end{bmatrix} \Bigg\} $ is equal. A linear mapping from $\\R^{n} $ to $\\R^{m} $ can be called a linear transformation. To facilitate the understanding of linear transformation, the mapping $f$ can be understood as a function (transformation matrix), the process of inputting  a vector, after the effect of $f$, and then outputting a vector, that is, the vector is moving. That is, when $\begin{bmatrix} x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix} $ (input) by $m \times n$ matrix $\begin{bmatrix} a_{11} & a_{12}& \ldots & a_{1n} \\ a_{21} & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2}& \ldots & a_{mn} \end{bmatrix} $ corrosponding from $\\R^{n} $ to $\\R^{m} $ linear mapping, formed image is $\begin{bmatrix} y_{1} \\y_{2}\\ \vdots \\y_{n} \end{bmatrix} $(output), whose matrix calculation formula is expressed as : $f\Bigg\{\begin{bmatrix} x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix}\Bigg\}=\begin{bmatrix} a_{11} & a_{12}& \ldots & a_{1n} \\ a_{21} & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2}& \ldots & a_{mn} \end{bmatrix} \begin{bmatrix} x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix}=\begin{bmatrix} a_{11} & a_{12}& \ldots & a_{1n}  \\ a_{21} & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2}& \ldots & a_{mn} \end{bmatrix} \Bigg\{ x_{1} \begin{bmatrix}1 \\0\\ \vdots \\0 \end{bmatrix} +x_{2} \begin{bmatrix}0 \\1\\ \vdots \\0 \end{bmatrix} + \ldots +x_{n} \begin{bmatrix}0 \\0\\ \vdots \\1 \end{bmatrix} \Bigg\}$.  For the image, assuming $x_{i} $ is elements of the set $X$, by mapping $f$, the element of the set $Y$ that corresponds to $x_{i} $ is called the image of $x_{i} $ by mapping $f$. 
 
Define three-dimensional C vector space, unit vector is $\vec{i},\vec{j},\vec{k}$, define v_1 vector coefficient is a1,a2,a3=-1,2,0, where 0 can be omitted, that is, v_1=a1*i+a2*j=-1*i+2*j vector. Given the new vector collection v_2, v_3, by calculating the reduced row echelon form matrix, the vector is judged to be linear independence, therefore, taking this dataset as the base, the new vector space can be established, and the transformation matrix is constructed from the base v_2, v_3, the coefficient matrix of v_1 times the transformation matrix is v_1 in the new vector space with respect to the C vector space v_1_N.

```python
fig, ax=plt.subplots(figsize=(12,12))
ax=fig.add_subplot( projection='3d')

a1,a2=sympy.symbols(["a1","a2",])
a1=-1
a2=2

C=CoordSys3D('C')
i, j, k = C.base_vectors()
v_0=Vector.zero
v_1=a1*i+a2*j
vector_plot_3d(ax,C,v_0,v_1,color='red',label='v_1',arrow_length_ratio=0.1)

v_2=-1*i+3*j
vector_plot_3d(ax,C,v_0,v_2,color='blue',label='v_2',arrow_length_ratio=0.1)
v_3=2*i+0*j
vector_plot_3d(ax,C,v_0,v_3,color='blue',label='v_3',arrow_length_ratio=0.1)

def vector2matrix_rref(v_list,C):
    '''
    function - The vector set is converted to a vector matrix, and the reduced row echelon form matrix is calculated.
    
    Paras:
    v_list - Vector list
    C - The coordinate system defined by Sympy
    
    return:
    v_matrix.T - The transformed vector matrix, namely the linear transformation matrix
    '''
    v_matrix=v_list[0].to_matrix(C)
    for v in v_list[1:]:
        v_temp=v.to_matrix(C)
        v_matrix=v_matrix.col_insert(-1,v_temp)
        
    print("_"*50)
    pprint(v_matrix.T.rref())
    return v_matrix.T
print("v_2,v_3 reduced row echelon form matrix:")
v_2_3=vector2matrix_rref([v_2,v_3],C)  #The vector set v_2,v_3 are judged to be vector independent by the reduced row echelon form matrix returned by rref(). Returns the linear transformation matrix


#From the vector set v_2,v_3 that are independent of each other, a new vector space is generated, concerning the original vector space, i, j, k unit vector, i, j, k is a new position.
v_1_N_matrix=Matrix([a1,a2]).T*v_2_3 #According to the transformation matrix, calculate the position of the vector a1,a2 with the same coefficient.
from sympy.vector import matrix_to_vector
v_1_N=matrix_to_vector(v_1_N_matrix,C) #Convert vector to matrix
vector_plot_3d(ax,C,v_0,v_1_N,color='orange',label='v_1_N',arrow_length_ratio=0.1) #Draw a vector with a new position

ax.set_xlim3d(-7,2)
ax.set_ylim3d(-2,7)
ax.set_zlim3d(-2,5)

ax.legend()
ax.view_init(90,0) #The angle of the graph can be rotated for easy observation
plt.show()
```

    v_2,v_3 reduced row echelon form matrix:
    __________________________________________________
    ⎛⎡1  0  0⎤        ⎞
    ⎜⎢       ⎥, (0, 1)⎟
    ⎝⎣0  1  0⎦        ⎠
    


<a href=""><img src="./imgs/9_4.png" height="auto" width="auto" title="caDesign"></a>


#### 1.3.2 Special linear mapping (linear transformation)
The matrix calculation method of linear mapping obtains new objects in the same vector space by multiplying the objects to be transformed(vector set) by the transformation matrix. It is convenient to use the Sympy matrix operation with a small data size and easy to observe matrix form. However, if the amount of data is large, use the matrix calculation method provided by the Numpy library. In the following case, download the city-point cloud data 'bildstein_station1' from [semantic3d](http://semantic3d.net/), and de-sample it to speed up the computation. The processing of point cloud data can refer to the point cloud part.

After the point cloud data is read, the information included has some coordinates and point color. Point coordinates can be understood as vectors in the $C$ vector space with unit vectors of $\vec{i},\vec{j},\vec{k}$; if the corresponding transformation matrix is provided, spatial transformation can be performed on the origin cloud data, such as movement, rotation, scaling, perspective, mirror image, shear, etc. The matrix corresponding to scaling is: $ \begin{bmatrix}scale & 0&0 \\0 &scale&0\\0&0&scale \end{bmatrix} $；The matrix corresponding to the rotation along the z-axis($\vec{k}$) is: $ \begin{bmatrix} cos( \theta ) & -sin( \theta )&0 \\sin( \theta ) &cos( \theta )&0\\0&0&1\end{bmatrix} $；The perspective of S(x,y,z) at the given point is: $ \frac{1}{- s_{z} }  \begin{bmatrix}  s_{z} & 0& s_{x}&0 \\0 &-s_{z} & s_{y} &0\\ 0&0&1&0 
 \\0&0&1&-s_{z} \end{bmatrix} $，Homogeneous coordinates have been used in the perspective transformation.


```python
import open3d as o3d
import numpy as np
import pandas as pd
cloud_pts_fp=r".\data\bildstein_station1.txt"
cloud_pts=pd.read_csv(cloud_pts_fp)
#cloud_pts['hex']=cloud_pts.apply(lambda row:'%02x%02x%02x' % (int(row.r/255),int(row.g/255),int(row.b/255)),axis=1)
print("data reading completed")
cloud_pts['hex']=cloud_pts.apply(lambda row:matplotlib.colors.to_hex([row.r/255,row.g/255,row.b/255]),axis=1) #Converts the color RGB format to HEX for Matplotlib graphic printing
```

    data reading completed
    


```python
import matplotlib.pyplot as plt
import matplotlib
import sympy,math
from sympy import Matrix,pprint
import numpy as np
fig, axs=plt.subplots(ncols=2, nrows=2,figsize=(20,20))
axs[0,0]=fig.add_subplot(2,2,1, projection='3d')
axs[0,1]=fig.add_subplot(2,2,2, projection='3d')
axs[1,0]=fig.add_subplot(2,2,3, projection='3d')
axs[1,1]=fig.add_subplot(2,2,4, projection='3d')

axs[0,0].scatter(cloud_pts.x,cloud_pts.y,cloud_pts.z,c=cloud_pts.hex,s=0.1)
#A - scaling
scale=0.5
f_scale=np.array([[scale,0,0],[0,scale,0],[0,0,scale]])
pts_scale=np.matmul(pts,f_scale)
axs[0,1].scatter(pts_scale[:,0],pts_scale[:,1],pts_scale[:,2],c=cloud_pts.hex,s=0.1)

#B - rotation
angle=math.radians(-20) #Convert degree to radians
f_rotate=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
pts_rotate=np.matmul(pts,f_rotate)
axs[1,0].scatter(pts_rotate[:,0],pts_rotate[:,1],pts_rotate[:,2],c=cloud_pts.hex,s=0.1)

#C - perspective
s1,s2,s3=2,2,2
f_persp=np.array([[-s3,0,0,0],[0,-s3,0,0],[s1,s2,0,1],[0,0,0,-s3]])*1/(1-s3)
pts_homogeneousCoordinates=np.hstack((pts,np.ones(pts.shape[0]).reshape(-1,1)))
pts_persp=np.matmul(pts_homogeneousCoordinates,f_persp)
axs[1,1].scatter(pts_persp[:,0],pts_persp[:,1],pts_persp[:,2],c=cloud_pts.hex,s=0.1)



axs[0,0].set_xlim3d(0,50)
axs[0,0].set_ylim3d(0,100)
axs[0,0].set_zlim3d(0,40)

axs[0,1].set_xlim3d(0,50)
axs[0,1].set_ylim3d(0,100)
axs[0,1].set_zlim3d(0,40)

axs[1,0].set_xlim3d(0,50)
axs[1,0].set_ylim3d(0,100)
axs[1,0].set_zlim3d(0,40)

axs[0,0].view_init(45,0) #The angle of the graph can be rotated for easy observation
axs[0,1].view_init(45,0)
axs[1,0].view_init(45,0)
axs[1,1].view_init(90,0)
plt.show()
```


<a href=""><img src="./imgs/9_5.png" height="auto" width="auto" title="caDesign"></a>


#### 1.3.3 Rank
$f$ is the mapping from $\\R^{n} $ to $\\R^{m} $, a collection of all elements that are mapped to zero elements ($\vec 0$), namely set $\Bigg\{\begin{bmatrix}  x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix} \bigg\rvert_{} \begin{bmatrix}0 \\0\\ \vdots \\0\end{bmatrix} =\begin{bmatrix} a_{11}  & a_{12}& \ldots & a_{1n}  \\ a_{21}  & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\  a_{m1}  & a_{m2}& \ldots & a_{mn} \end{bmatrix} \begin{bmatrix}  x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix}\Bigg\}$ is called the kernel of the mapping $f$, and is generally expressed as $K_{er} f$。To correspond to the kernel of $f$, then, the value domain of mapping $f$, namely set $f\Bigg\{\begin{bmatrix} y_{1}  \\y_{2} \\ \vdots \\y_{m} \end{bmatrix}   \bigg\rvert_{}   \begin{bmatrix}  y_{1}  \\y_{2} \\ \vdots \\y_{m} \end{bmatrix} =\begin{bmatrix} a_{11}  & a_{12}& \ldots & a_{1n}  \\ a_{21}  & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\  a_{m1}  & a_{m2}& \ldots & a_{mn} \end{bmatrix} \begin{bmatrix}  x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix}\Bigg\}$ is called the image space of mapping $f$, and is generally expressed as $I_{m} f$。
 
 $K_{er} f$ is the subspace of $\\R^{n} $, $I_{m} f$ is the subspace of $\\R^{m} $. Between $K_{er} f$ and $I_{m} f$, there is 'dimension formula': $n- dimK_{er} f=dim I_{m}f $。
 
 The number of linearly independent vectors in vector $ \begin{bmatrix} a_{11} \\ a_{21}\\ \vdots \\ a_{m1}   \end{bmatrix},\begin{bmatrix} a_{12} \\ a_{22}\\ \vdots \\ a_{m2}   \end{bmatrix} , \ldots , \begin{bmatrix} a_{1n} \\ a_{2n}\\ \vdots \\ a_{mn}   \end{bmatrix} $,namely the dimension of $I_{m} f$ subspace of $\\R^{m} $ is called the rank of the $m \times n$ matrix $\begin{bmatrix} a_{11}  & a_{12}& \ldots & a_{1n}  \\ a_{21}  & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\  a_{m1}  & a_{m2}& \ldots & a_{mn} \end{bmatrix} $，which is generally expressed as $rank\begin{bmatrix} a_{11}  & a_{12}& \ldots & a_{1n}  \\ a_{21}  & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\  a_{m1}  & a_{m2}& \ldots & a_{mn} \end{bmatrix} $,或$r(A),rank(A),rk(A)$。
 
 matrix.rank() provided by Sympy is used to calculate rank.

### 1.4 Eigenvalues and eigenvectors
In the special linear mapping section, the given unit vector is $\vec{i},\vec{j},\vec{k}$, that is, the $C$ vector space of  $\begin{bmatrix}1& 0&0 \\0 & 1&0 \\0&0&1\end{bmatrix} $ , by the transformation matrix $ \begin{bmatrix}scale1 & 0&0 \\0 &scale2&0\\0&0&scale3 \end{bmatrix} $, the original vector set(point cloud data) is scaled. For the convenience of explanation, if only consider a point in the point cloud data, $c_{1}, c_{2},c_{3}=x,y,z=39,73,22$, the point in the $C$ vector space is a vector $C_{1}i+C_{2}j+C_{3}k=C_{1} \begin{bmatrix}1\\0\\0 \end{bmatrix}+ C_{2} \begin{bmatrix}0\\1\\0 \end{bmatrix}+ C_{3} \begin{bmatrix}0\\0\\1 \end{bmatrix}$. Doing linear mapping and formula transformation,$ \begin{bmatrix}scale1 & 0&0 \\0 &scale2&0\\0&0&scale3 \end{bmatrix}\Bigg\{C_{1} \begin{bmatrix}1\\0\\0 \end{bmatrix}+ C_{2} \begin{bmatrix}0\\1\\0 \end{bmatrix}+ C_{3} \begin{bmatrix}0\\0\\1 \end{bmatrix}\Bigg\}= C_{1} \begin{bmatrix}scale1 & 0&0 \\0 &scale2&0\\0&0&scale3 \end{bmatrix}\begin{bmatrix}1\\0\\0 \end{bmatrix}+C_{2} \begin{bmatrix}scale1 & 0&0 \\0 &scale2&0\\0&0&scale3 \end{bmatrix}\begin{bmatrix}0\\1\\0 \end{bmatrix}+C_{3} \begin{bmatrix}scale1 & 0&0 \\0 &scale2&0\\0&0&scale3 \end{bmatrix}\begin{bmatrix}0\\0\\1 \end{bmatrix}=C_{1}\begin{bmatrix}scale1\\0\\0\end{bmatrix}+C_{1}\begin{bmatrix}0\\scale2\\0\end{bmatrix}+C_{1}\begin{bmatrix}0\\0\\scale3 \end{bmatrix}=C_{1}\Bigg\{scale1 \begin{bmatrix}1\\0\\0 \end{bmatrix}  \Bigg\}+C_{2}\Bigg\{scale2 \begin{bmatrix}0\\1\\0 \end{bmatrix}  \Bigg\}+C_{3}\Bigg\{scale3 \begin{bmatrix}0\\0\\1\end{bmatrix}  \Bigg\}$，The transformed formula remains the unit vector $\vec{i},\vec{j},\vec{k}$ unchanged in the $C$ vector space, then  $scale1,scale2,scale3$ are the eigenvalues of the transformation matrix, and $\begin{bmatrix}1\\0\\0\end{bmatrix} $ corresponding to $scale1$, $\begin{bmatrix}0\\1\\0\end{bmatrix} $ corresponding to $scale2$, $\begin{bmatrix}0\\0\\1\end{bmatrix} $ corresponding to $scale3$ are its eigenvectors.

When $\begin{bmatrix} x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix} $ (input) passes the $m \times n$ matrix $\begin{bmatrix} a_{11} & a_{12}& \ldots & a_{1n} \\ a_{21} & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2}& \ldots & a_{mn} \end{bmatrix} $, the corresponding linear mapping of $f$ from $\\R^{n} $ to $\\R^{m} $, the formed image is $\begin{bmatrix} y_{1} \\y_{2}\\ \vdots \\y_{n} \end{bmatrix} = \lambda \begin{bmatrix} x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix} $(output).  $\lambda$ is called eigenvalue of the square matrix $\begin{bmatrix} a_{11} & a_{12}& \ldots & a_{1n} \\ a_{21} & a_{22}& \ldots & a_{2n} \\ 
 \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2}& \ldots & a_{mn} \end{bmatrix} $, $\begin{bmatrix} x_{1} \\x_{2}\\ \vdots \\x_{n} \end{bmatrix} $ is called the eigenvector corresponding to the eigenvalue $\lambda$. Furthermore, the zero vector cannot be interpreted as an eigenvector.

 
The eigenvalue and the eigenvector are directly calculated utilizing Sympy library, matrix.eigenvals()，matrix.eigenvects().


```python
scale1,scale2,scale3=sympy.symbols(['scale1','scale2','scale3'])
M=Matrix(3, 3, [scale1*1, 0, 0, 0, scale2*1, 0, 0, 0, scale3*1])
pprint(M)
print("eigenvals:")
pprint(M.eigenvals())
print("eigenvects:")
pprint(M.eigenvects())
```

    ⎡scale₁    0       0   ⎤
    ⎢                      ⎥
    ⎢  0     scale₂    0   ⎥
    ⎢                      ⎥
    ⎣  0       0     scale₃⎦
    eigenvals:
    {scale₁: 1, scale₂: 1, scale₃: 1}
    eigenvects:
    ⎡⎛           ⎡⎡1⎤⎤⎞  ⎛           ⎡⎡0⎤⎤⎞  ⎛           ⎡⎡0⎤⎤⎞⎤
    ⎢⎜           ⎢⎢ ⎥⎥⎟  ⎜           ⎢⎢ ⎥⎥⎟  ⎜           ⎢⎢ ⎥⎥⎟⎥
    ⎢⎜scale₁, 1, ⎢⎢0⎥⎥⎟, ⎜scale₂, 1, ⎢⎢1⎥⎥⎟, ⎜scale₃, 1, ⎢⎢0⎥⎥⎟⎥
    ⎢⎜           ⎢⎢ ⎥⎥⎟  ⎜           ⎢⎢ ⎥⎥⎟  ⎜           ⎢⎢ ⎥⎥⎟⎥
    ⎣⎝           ⎣⎣0⎦⎦⎠  ⎝           ⎣⎣0⎦⎦⎠  ⎝           ⎣⎣1⎦⎦⎠⎦
    

### 1.5 key point
#### 1.5.1 data processing technique

* Use the Sympy library to calculate the matrix and vector.

* Numpy library for computing matrices(especially large volumes of data)

* Use Open3D to process point cloud data.

#### 1.5.2 The newly created function tool

* funciton - Transform vector and Matrix data formats in Sympy to Matplotlib printable  data formats, `vector_plot_3d(ax_3d,C,origin_vector,vector,color='r',label='vector',arrow_length_ratio=0.1)`

* function - Given the vector and the corresponding coefficient, draw along the vector, `move_alongVectors(vector_list,coeffi_list,C,ax,)`

* function - The vector set is converted to a vector matrix, and the reduced row echelon form matrix is calculated,`vector2matrix_rref(v_list,C)`

#### 1.5.3 The python libraries that are being imported


```python
import sympy
from sympy import Matrix,pprint
import matplotlib.pyplot as plt
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Vector, BaseVector
from sympy.vector import Vector
from sympy.vector import CoordSys3D
from sympy.abc import a, b, c,d,e,f,g,h,o,p 
from sympy import pprint,Eq,solve
from sympy import solve_linear_system
from sympy.vector import matrix_to_vector

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import PathPatch,Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

import open3d as o3d
import numpy as np
import pandas as pd
import math
```

#### 1.5.4 Reference
1. Shin Takahashi, Iroha Inoue.The Manga Guide to Linear Algebra.No Starch Press; 1st Edition (May 1, 2012);
2. Gilbert Strang.Introduction to linear algbra[M].Wellesley-Cambridge Press; Fifth Edition edition (June 10, 2016)
