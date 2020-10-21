# Code tool index
> This book's purpose is not only the textbook in the general sense but also the description of relevant experiments in the specialty. Simultaneously, many code fragments compiled by experiments can form a tool that can be easily transferred and used to form a toolkit.

## 1. Baidu Map POI data crawler and geospatial point map
1. function-Baidu map open platform POI data crawler, `baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False)`;
2. function-Convert .csv format of POI data to DataFrame of pandas, `csv2df(poi_fn_csv)`；

## 2. Multiple classification POI data crawling and Descriptive statistics
3. function-Baidu Map open platform POI data mass crawling，`baiduPOI_batchCrawler(poi_config_para)`。need to call the single POI classification crawler function. `baiduPOI_dataCrawler(query_dic,bound_coordinate,partition,page_num_range,poi_fn_list=False)`
4. funciton-With the folder path as the key, the value is a list of all filenames under the folder. File types can be specified.  `filePath_extraction(dirpath,fileType)`
5. funciton-.csv format POI data is converted to GeoDataFrame in batch，`poi_csv2GeoDF_batch(poi_paths,fields_extraction,save_path)`。which needs to call .csv to DataFrame data format provided by pandas function  `csv2df(poi_fn_csv)`
6. funciton-use Plotly to display data in DataFrame format as a table, `ployly_table(df,column_extraction)`
7. function-frequency calculation，`frequency_bins(df,bins)`

## 3. Normal distribution and probability density function, outlier handling
8. funciton-dataset(observed/empirical data) compared with the standard normal distribution(theoretical set)，`comparisonOFdistribution(df,field,bins=100)`
9. function-Judge outliers，`is_outlier(data,threshold=3.5)`

## 4. OpenStreetMap（OSM）data processing
10. function-Convert the polygon of the shape to the polygon data format of osmium (.txt) for cropping the .osm map data.`shpPolygon2OsmosisTxt(shape_polygon_fp,osmosis_txt_fp)`
11. class-Read the .osm data by inheriting the osmium class osmium.SimpleHandler, `osmHandler(osm.SimpleHandler)`
12. function-Save the OSM data one by one, depending on the condition.（node, way and area）,`save_osm(osm_handler,osm_type,save_path=r"./data/",fileType="GPKG")`
13. function-Calculate the current time, `start_time()`
14. function-Calculate the duration, `duration(start_time)`

## 5.Kernel density estimation and geospatial point density distribution
15. function - Convert point data in .shp format to .tif raster, `pts2raster(pts_shp,raster_path,cellSize,field_name=False)`
16. function -  Convert point data in GeoDaraFrame format to raster data, `pts_geoDF2raster(pts_geoDF,raster_path,cellSize,scale)`

## 6.Standard error, central limit theorem, Student's t-distribution, statistical significance, effect size, confidence interval; Geospatial distribution and correlation analysis of public health data
17. function -  Print DataFrame format data in Jupyter as HTML, `print_html(df,row_numbers=5`
18. function - Generate a color list or dictionary,  `generate_colors()`

## 7.Simple regression, multiple regression
19. function - Draw the connection line in the subgraph of the Matplotlib, `demo_con_style(a_coordi,b_coordi,ax,connectionstyle)`
20. function - The determination coefficient of the regression equation, `coefficient_of_determination(observed_vals,predicted_vals)`
21. function - Simple linear regression equation - regression significance test(regression coefficient test), `ANOVA(observed_vals,predicted_vals,df_reg,df_res)`
22. function - The simple linear regression confidence interval estimation, and the predicted interval, `confidenceInterval_estimator_LR(x,sample_num,X,y,model,confidence=0.05)`
23. function - DataFrame format data, group calculated the Pearson correlation coefficient,`correlationAnalysis_multivarialbe(df)`
24. function - The non-modified degree determination coefficient of the regression equation, `coefficient_of_determination_correction(observed_vals,predicted_vals,independent_variable_n)`
25. function - Multiple linear regression equation - regression significance test(regression coefficient test), the population test of all regression coefficient, and single regression coefficient tests, `ANOVA_multivarialbe(observed_vals,predicted_vals,independent_variable_n,a_i,X)`
26. function - Multiple linear regression confidence interval estimation, and predicted interval, `confidenceInterval_estimator_LR_multivariable(X,y,model,confidence=0.05)`

## 8. Regression of public health data, with gradient descent method
27. function - Returns the nearest point coordinates for a specified number of neighbors,`k_neighbors_entire(xy,k=3)`
28. function - Polynomial regression degree selection and regularization, `PolynomialFeatures_regularization(X,y,regularization='linear')`
29. Gradient descent method - define the model, define the loss function, define the gradient descent function, define the training model function

## 9. A code representation of the linear algebra basis
30. funciton - Transform vector and Matrix data formats in Sympy to Matplotlib printable  data formats, `vector_plot_3d(ax_3d,C,origin_vector,vector,color='r',label='vector',arrow_length_ratio=0.1)`
31. function - Given the vector and the corresponding coefficient, draw along the vector, `move_alongVectors(vector_list,coeffi_list,C,ax,)`
32. function - The vector set is converted to a vector matrix, and the reduced row echelon form matrix is calculated,`vector2matrix_rref(v_list,C)`

## 10. Landsat remote sensing image processing, digital elevation, principal component analysis
33. function - Sort the list of files by the number in the file name, `fp_sort(fp_list,str_pattern,prefix="")`
34. function - Read 'landsat *_MTL.txt' file and extract the required information. `LandsatMTL_info(fp)`
35. funciton - Transform vector and matrix data formats in Sympy to Matplotlib printable data formats, `vector_plot_2d(ax_2d,C,origin_vector,vector,color='r',label='vector',width=0.022)`
36. function - Given the center of the circle, the radius, the division number, and calculate the head and tail coordinates of all diameters. `circle_lines(center,radius,division)`
37. function - Calculate the projection of a point in two dimensions onto a line, `point_Proj2Line(line_endpts,point)`
38. function -  Calculate the NDVI index, `NDVI(RED_band,NIR_band)`

## 11.Remote sensing image interpretation (based on NDVI), the establishment of sampling tool(GUI_tkinter), confusion matrix
39. function - Given cutting boundaries, batch cropping raster data, `raster_clip(raster_fp,clip_boundary_fp,save_path)`
40. function - Specify the band and display multiple remote sensing images at the same time, `bands_show(img_stack_list,band_num)`
41. function -Converts a variable name to a string,  `variable_name(var)`
42. function - contract stretching images,`image_exposure(img_bands,percentile=(2,98))`
43. function - Divide the data by a given percentile and give a fixed value, integer, or RGB color value,`data_division(data,division,right=True)`
44. function - Multiple raster data, given percentile, observe changes, `percentile_slider(season_dic)`
45. Based on Tkinter, an interactive GUI sampling tool is developed

## 12. Point cloud data(Lidar) processing——classified data, DSM, building height extraction, interpolation
46. function - Convert single .las point cloud data into classification raster data and DSM raster data, etc, `las_info_extraction(las_fp,json_combo)`
47. function - Displays the classification raster file generated by the .las file and displays the legend, `las_classification_plotWithLegend(las_fp)`
48. function - Batch conversion .las point cloud data into DSM and classification raster, `las_info_extraction_combo(las_dirPath,json_combo_)`
49. function - Merge multiple rasters into one, `raster_mosaic(dir_path,out_fp,)`
50. function - Transfer the function provided by Rasterio that defines the data type with the minimum array size, `get_minimum_int_dtype(values)`
51. function -  Gets the projected coordinates of the given raster-crs, `get_crs_raster(raster_fp)`
52. function - Transform raster projection,`raster_reprojection(raster_fp,dst_crs,save_path)`
53. function - Crop data in .shp format to the extent of a given raster and define a projection as the same as the raster, `clip_shp_withRasterExtent(vector_shp_fp,reference_raster_fp,save_path)`
54. function - The interpolation method 'rasterio.fill' is used to complete the missing data,`rasters_interpolation(raster_path,save_path,max_search_distance=400,smoothing_iteration=0)`
55. function - Display remote sensing images(one band) using the Earthpy library,`raster_show(raster_fp,title='raster',vmin_vmax=[0.25,0.95],cmap="turbo")`

## 13. Convolution, SIR propagation model, cost raster and species dispersal, SIR spatial propagation model
56. class - One dimensional convolution animation analysis to customize the system function and signal function, `class dim1_convolution_SubplotAnimation(animation.TimedAnimation)`
57. function - Define system response functions. Type-1, `G_T_type_1()`
58. function - Define the input signal function. Type-1, `F_T_type_1(timing)`
59. function - Define system response functions. Type-2, `G_T_type_2()`
60. function - Define the input signal function. Type-2, `F_T_type_2(timing)`
61. function - Read Matlab chart data, Type-A,  `read_MatLabFig_type_A(matLabFig_fp,plot=True)`
62. function - One dimensional convolution is applied to segment the data according to the jump point. `curve_segmentation_1DConvolution(data,threshold=1)`
63. function - Split the list according to the index,`lst_index_split(lst, args)`
64. function - 'Flattening list function,  `flatten_lst=lambda lst: [m for n_lst in lst for m in flatten_lst(n_lst)] if type(lst) is list else [lst]`
65. function - Nested list, sublists interpolated before and after, `nestedlst_insert(nestedlst)`
66. function - Use the method provided by Matplotlib to return floating-point RGB randomly, `uniqueish_color()`
67. function - Define the SIR model, the differential function,  `SIR_deriv(y,t,N,beta,gamma,plot=False)`
68. function - Displays the image as well as the color R-value, or G, B value, `img_struc_show(img_fp,val='R',figsize=(7,7))`
69. class - 2 D convolution diffusion is defined based on the SIR model.`class convolution_diffusion_img`
70. function - Read in .gif and display dynamically, `animated_gif_show(gif_fp,figsize=(8,8))`
71. fuction - Down-sampling a two-dimensional array, according to the maximum value of the value frequency in each block, the value appearing at most is the sampling value of each block., `downsampling_blockFreqency(array_2d,blocksize=[10,10])`
72. class - the SIR spatial propagation model, `SIR_spatialPropagating`
