# OilGasFlaringFromVNF
This program read daily flaring activity, volume of gas flared and process the data to determine how LNG export facilities are operating and how often they flare. This program is designed extract flaring records from VNF dataset using geopandas. It uses cKDTree function to cluster flare points to the nearest facilities' stack locations.

Features include:
1. The capability of working with ee.FeatureCollection("EDF/OGIM/current") dataset is added.
2. Various probability functions, including Gumbel-GEV-LogNormal-GPD-Exponential, are added.
3. Improved and Faster version of clustering daily flares around facilities.
4. Various classes are designed to split the program to complete different tasks.

Three classes of VNF, ProbClass, and PloterClass are provided in vnf_data_processing, probability_theory, ploter_functions to improve the code's performance and readability.
The main script is process-vnf.py, which import all classes and use them.

## Usage
Users can use this program without installing the requiring libraries by by placing the '.so' files within the main scripts. Within the main script (similar to the process-vnf.py script), users can filter and process data for all Export LNG facilities.

Users can also modify the codes and functions as required within their application on their side under the GNU AGPL-3.0 Licence and publish their code under the same licence (if they wanted).

First, download the flaring records (placed them in the 'vnf' folder) from https://eogdata.mines.edu/products/vnf/ and volume of gas flared (placed them the 'gas' folder) from https://eogdata.mines.edu/products/vnf/global_gas_flare.html. Then, the program can be run. An example of these records are provided.

## Programs
The main program and functions are grouped into three scripts. Each call needs to be loaded within the main script. Then, the functions can be used to process and plot data. The three main codes are
1. ploter_functions
2. probability_theory
3. vnf_data_processing

### ploter_functions.py
ploter_functions contains a class with all required functions to plot the results.

### probability_theory.py
probability_theory contains a class with all data processing funtions including, fitting to 5 probability theory distributions, goodness of fit, and data processing, and data handling. 

### vnf_data_processing.py
vnf_data_processing contains a class with all required functions to read and process the VNF records.

## Required Python Libraries
### Data processing
1. pandas
2. numpy
3. geopandas
4. shapely
5. ast
6. scipy

### Working with Google Earth Engine
1. geemap
2. ee

### Ploting the results
1. matplotlib
2. cartopy
3. seaborn
