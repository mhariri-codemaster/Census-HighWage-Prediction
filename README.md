# Working on the Imbalanced Census Data

The Census dataset can be found here:  
http://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/  
This project was inspired from the article at:  
https://www.analyticsvidhya.com/blog/2016/09/this-machine-learning-project-on-imbalanced-data-can-add-value-to-your-resume/  
A better representation of the train and test data is also included on that webpage which is the one assumed to be used by the code here. To use the code, download the datasets for train and test and place them in the respective folders.

The idea here is to use the US Census data to predict whether the income of the individuals in the given test dataset is greater than $50000. The test dataset is labeled as well. This dataset is heavily imbalanced with most of the records (about 93%) labeled below the $50000 mark. The data is also not suitable in its raw form for machine learning as it is missing many values and contains unnecessary information which might make the job harder. Our implementation fixes those problems to create new modified datasets which are then used to train Neural Nets to predict the income level of the individuals.


## Approach

The dataset is about medium sized. Nonetheless, the implementation assumes that the dataset is large and operates on the data in chunks. That way, the programs could be used on any larger sized datasets. 
The data is modified to drop some features that are redundant or not informative enough. Missing values are also added for some features using pivot tables. The data is finally encoded so that categorical features are tranformed into numerical ones. We provide two types of encoding, one hot coding and binary coding. 
If all predictions returned a label of 0 (income<$50000) then about 93% of the data will be labeled correctly. However, 100% of the minority results (income>$50000) will be labeled incorrectly. To overcome this problem, the simple apporach is to oversample the minority data or to undersample the majority data. We provide two functions that do exactly that. To check the efficiency of these approaches, we also look at the fraction of majority and minority tests that have bee labeled correctly. 
Feel free to try different models in the "DataModel.py" code.

## Directory

The main directory contains the following files/folders:
-train 
--train.csv 
-test 
--test.csv 
-src
--DataModels.py 
--DataModelsFunc.py 
--DataPrep.py 
--DataPrepFunc.py 
--DataPrepClasses.py 
-README.md
-run.sh (used to run on the terminal)

### File.Folder Description

* **train:** It has the training file "train.csv" 
* **test:** It has the test file "test.csv"
* **src:** It has the Python code files "DataPrep.py" and "DataModels.py". Thew first is used to manipulate the raw data, encode it and create the modified new files for train or test in their respective folders while the second is used to train the models using the modified datasets. In addition, the files "DataPrepFunc.py", "DataPrepClasses.py" and "DataModelsFunc.py" contain all the user written dependencies for these functions. 
* **README.md**: It is the readme file.
* **run.sh**: It is used to run the code from shell terminal. To run, simply type ./run.sh.

## Prerequisites

The code is written in Python 3.5.2

### Additional Libraries/Depndencies

I used pandas to prepare the data and sklearn for the learning part. The following libraries are required: 
-numpy
-pandas (for data manipulation)
-os
-sklearn (for the Neural Nets)
-pickle (for saving the encoders)
-matplotlib.pyplot (for plotting the scores)

If Python 3.1 or 2.7 or higher are used, no installations are required

## Authors

* **Mohamed Hariri Nokob**
