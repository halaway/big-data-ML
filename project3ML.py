from pyexpat import features
from random import random
import pandas as pd
from pyspark import SparkContext, SparkConf
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from pyspark.sql import SQLContext
from sklearn.metrics import roc_curve, auc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import col
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline




from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder




"""
    Machine Learning Model Using Spark mllib
"""



class ML:
    """Creating Machine Learning Class using Spark.
    """
    ##
    # 
    # @param file_name: Name of the File used for loading data
    # #
    def __init__(self, file_name):
        """Loading CSV File and using Spark for creating Clusters.
        Initializing ML Class.
        """
        # Create a SparkConf object to set up configuration
        conf = SparkConf().setAppName("SparkContextExample").setMaster("local[*]")\
            .set("spark.sql.debug.maxToStringFields", 1000)

        # SparkContext object with configuration
        sc = SparkContext(conf=conf)

        # Creating SQL Context 
        sql_c = SQLContext(sc)

        # Creating a SQL DataFrame
        df = sql_c.read.csv(file_name, header=True, inferSchema=True)

        # Setting objects dataframe with loaded data
        self.data_total = df

        #self.sc = sc


    ##
    # Selecting Features needed for Training our Models
    # 
    # @param Target: Target Label being used for model
    # #
    def feature_selection(self, target='diagnosis'):

        # Selecting the target variables used for training
        target_values = self.data_total.select(target)


        # Selecting Feature values used for training 
        feature_values = self.data_total.select([i for i in self.data_total.columns\
            if i not in {'diagnosis','id'}])

        # Setting Object Target Value
        self.target_values = target_values

        # Setting Object Feature Values
        self.features_values = feature_values


    ##
    # 
    # @return Model
    # #
    def random_forest(self):
        """Extracting the independent and target colums.
        Transforming the feature columns into a single dimensional vector 
        used for making boolean classifications for target values based on "B" or "M".
        Create predictions from our data after an 80/20 split, later stored
        in the ML Object for training a SVC or RandomForest.
        """
        # Creating a transforming for converting feature columns
        vect_assembler = VectorAssembler(inputCols = self.features_values.columns, \
            outputCol='features')
        # Transforming total loaded data
        features = vect_assembler.transform(self.data_total)
        # Selecting our Feature and Targt
        df = features.select('features','diagnosis')
        # Setting Object split Data
        self.split_data = df

        # Transforming our diagnosis lables as either 1.0 or 0.0
        indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
        indexed_df = indexer.fit(df).transform(df)

        # Splitting data into train and test sets
        train_df, test_df = indexed_df.randomSplit([0.8, 0.2], seed=42)
        # Creating Model
        random_classifier = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
        # Fiting the diagnosis labeled data 
        model = random_classifier.fit(train_df)

        # Make predictions on the test data
        predictions = model.transform(test_df)
        # Setting Our objects Random Forest Predictions
        self.predictions = predictions


        return 



    ##
    # Plotting perfomance charts of our Model Predictions
    # 
    # @return: ROC Curve
    # #
    def plot_performance(self, classifier='RF'):
        """Uses Model's predicted scores for plotting the ROC as
        a measure of our model's performance
        """
        # Setting predictions from Model's performance
        predictions = self.predictions

        # Getting values from our probability columns
        y_score = np.array(predictions.select('probability').\
            rdd.map(lambda row: row['probability'][1]).collect())
    
        # Getting the true values from predicted or labeled column
        y_true = np.array(predictions.select('label').rdd.map(lambda row: row['label']).collect())

        # Computing ROC curve 
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plotting ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()


    ##
    # Plotting Confusion Matrix Based on Model Predictions
    # 
    # @return: Plots a confusion matrix between predicted and
    # actual values
    # #
    def plot_confusion_matrix(self):
        """Using the model's predictions on test data, 
        the functon plots the confusion matrix using the
        MutiClassMetrics function
        """

        predictions_mapped = self.predictions.select("prediction", "label")\
            .rdd.map(lambda row: (float(row.prediction), float(row.label)))

        #Creating the MulticlassMetrics object
        metrics = MulticlassMetrics(predictions_mapped)

        #Getting the confusion matrix
        conf_matrix = metrics.confusionMatrix().toArray()


        #Plotting Confusion Matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = [0, 1] 
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, str(int(conf_matrix[i][j])), horizontalalignment='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

        plt.show()



        return 
    

    ##
    # Creating Support Vector Machine Classifer
    # with cross validation of five folds 
    # 
    # #
    def support_vector_CV_fold(self):
        """Extracting the independent and target colums.
        Transforming the feature columns into single dimensional vector and make
        boolean classifications for target values based on "B" or "M".
        Create predictions from our data after an 80/20 split, later stored
        in the ML Object for training a SVC with cross validation of 5 folds.
        """

        # Creating a transforming for converting feature columns
        vect_assembler = VectorAssembler(inputCols = self.features_values.columns, \
            outputCol='features')

        # Transforming total loaded data
        features = vect_assembler.transform(self.data_total)

        # Selecting our Feature and Targt
        df = features.select('features','diagnosis')


        # Setting Object split Data
        self.split_data = df

        # Transforming our diagnosis lables as either 1.0 or 0.0
        indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
        indexed_df = indexer.fit(df).transform(df)

        # Splitting data into train and test sets
        train_df, test_df = indexed_df.randomSplit([0.8, 0.2], seed=42)

        model_svc = LinearSVC(featuresCol = 'features',labelCol="label", maxIter=50)

        # Creating a pipeline with the model
        pipeline = Pipeline(stages=[model_svc])

        # Set up the parameter grid for tuning
        paramGrid = ParamGridBuilder() \
            .addGrid(model_svc.regParam, [0.1, 0.01]) \
            .addGrid(model_svc.threshold, [0.0, 0.1]) \
            .build()

        evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label', metricName='areaUnderROC')

        # setting up cross validator
        crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,\
            evaluator=evaluator,numFolds=5)

        # training model using CrossValidator
        cvModel = crossval.fit(train_df)
        # Get the best model from CrossValidator
        best_model = cvModel.bestModel
        # creating predictions based on test data
        predictions = best_model.transform(test_df)
        # set predictions
        self.predictions = predictions

        predictions.show()

        return


    ##
    # Creating Support Vector Machine Classifer
    # 
    # @return: SVC with model predictions 
    # #
    def support_vector(self):
        """Extracting the independent and target colums.
        Transforming the feature columns into single dimensional vector and make
        boolean classifications for target values based on "B" or "M".
        Create predictions from our data after an 80/20 split, later stored
        in the ML Object for training a SVC or RandomForest.
        """
        # Creating a transforming for converting feature columns
        vect_assembler = VectorAssembler(inputCols = self.features_values.columns, \
            outputCol='features')

        # Transforming total loaded data
        features = vect_assembler.transform(self.data_total)

        # Selecting our Feature and Targt
        df = features.select('features','diagnosis')

        # Setting Object split Data
        self.split_data = df

        # Transforming our diagnosis lables as either 1.0 or 0.0
        indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
        indexed_df = indexer.fit(df).transform(df)

        # Splitting data into train and test sets
        train_df, test_df = indexed_df.randomSplit([0.8, 0.2], seed=42)

        train_df.show()

        model_svc = LinearSVC(featuresCol = 'features',labelCol="label", maxIter=50)
        model_svc = model_svc.fit(train_df)
        # Creating our model Predictions 
        predictions = model_svc.transform(test_df)
        # Setting up Our objects Random Forest Predictions
        self.predictions = predictions




    ##
    # Model Performance Metrics
    # Report using F1 score, precision, recall, and ROC curve
    # @return: Prints Evaluation Metrics 
    # #
    def model_performance(self):
        """Uses previous model Prediction scores to 
        return various performance metrics for our RFC model.
        Also uses binary classifier evaluators and metric counts.
        """

        # Getting Prediction values from previous prediction Models
        predictions = self.predictions

        # Binary Classfication 
        binary_evaluator = BinaryClassificationEvaluator(labelCol='label', \
            rawPredictionCol='prediction', metricName='areaUnderROC')

        #Calculate True Positive Scores
        true_positives = predictions.filter((col('label') == 1) & (col('prediction') == 1)).count()
        #Calculate False Positive Scores
        false_positives = predictions.filter((col('label') == 0) & (col('prediction') == 1)).count()
        #Calculate False Negatives
        false_negatives = predictions.filter((col('label') == 1) & (col('prediction') == 0)).count()
        #Calculate True Negatives
        true_negatives = predictions.filter((col('label') == 0) & (col('prediction') == 0)).count()
        

        #Computing the Recall | Precision | Accuracy | F1 Score
        recall = true_positives / (true_positives + false_negatives + 1e-10)  
        precision = true_positives / (true_positives + false_positives + 1e-10)
        accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

        #Printing AUC Score Using the Binary Evaluator
        auc_score = binary_evaluator.evaluate(predictions)
        print("Area under the Curve:", auc_score)

        #Printing the Precision, Recall, Accuracy, and F1 Score
        print("Precision:", precision)
        print("Recall:", recall)
        print("Accuracy:", accuracy)
        print("F1 Score:", f1_score)
        
        return 





def main():

    file = "project3_data.csv"

    # Creating ML Object using Spark
    data = ML(file_name=file)

    # Extracting Features and Target Values from Data Frame
    data.feature_selection()

    # Creating Linear Support Vector Classifier Model 
    data.support_vector()

    # Printing Model Metrics
    data.model_performance()


    # Creating ML Object using Spark
    #random_F = ML(file_name=file)

    # Extracting Features and Target Values from Data Frame
    #random_F.feature_selection()

    # Creating Random Forest Classifier Model 
    #random_F.random_forest()

    # Plotting Confusion Matrix for ML model
    #random_F.plot_confusion_matrix()

    # Plotting ROC Curve 
    #random_F.plot_performance()

    # Printing Model Metrics
    #random_F.model_performance()





    return 0


if __name__=='__main__':
    main()