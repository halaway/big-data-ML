# Machine Learning Models: Integrating Spark
### Introduction
Leveraging distributed systems for tackling machine learning problems, typically increases query optimization and model performance when accounting for faster execution times and parallel data processing.

We are given a CSV file consisting of 569 rows of individual samples defining some diagnosis of either "Benign” or ”Malignant” and twenty measured clinical variables. For this project, we use Spark along with two machine learning models, ie. LinearSVC and Random Forest Classifiers, for predicting the positive target variable of our data set, or the ”Malignant” diagnosis.

### Integrating Spark With LinearSVC and RandomForest
This project is essentially contained within the ML Class, where different methods
are used for instantiating, fitting, and plotting the model along with their performance
metrics. 

### Linear SVC Performance Metrics

<div style="display: flex; align-items: center;">
    <div style="flex: 1;">
        <img align='left' src = "https://github.com/halaway/big-data-ML/assets/31904474/33ebd981-cecf-4906-9091-e4d748f86f75" width=35% height=35% > 
    </div>
</div>
   
<p align='center'>
    
| Precision       | Recall  | Accuracy   | F1-Score  | 
| -------- |:------:| --------:| ---------------------- |
|0.999|   0.914    |	   0.965	  |  0.955     |  
</p>


Testing the model's predicted values against the true values created the following metrics 
as measures of the model's predictive ability. 
<br clear="left"/>


### Random Forest Performance Metrics
<img align='left' src= "https://github.com/halaway/big-data-ML/assets/31904474/4fd81ba1-9b36-4251-a38b-df103063720e" width=35% height=35% > 
<img align = 'center' src="https://github.com/halaway/big-data-ML/assets/31904474/2503670a-4817-45c0-b8c9-df5e045dc22a" width=35% height=35%>

| Precision       | Recall  | Accuracy   | F1-Score  | 
| -------- |:------:| --------:| ---------------------- |
|0.941|   0.914  |	  0.942	  |0.928 | 

<br clear="left"/>




# General Use
The main file contains a few lines of code that create an ML Class depending on 
the model type: LinearSVC or RandomForest.

A class is created like: 
  - support_svc = ML('file/path/project3.csv')
  - support_svc.feature_selection()
  - ...
### Note
Dealing with Support Vector Machines doesn't usually allow for plotting ROC curves 
since they don't often predict probabilities(I think )

