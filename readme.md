# Machine Learning Models: 
### Spark With LinearSVC and RandomForest
This project is essentially contained within the ML Class, where different methods
are used for instantiating, fitting, and plotting the model along with their performance
metrics. 

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

