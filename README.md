# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree. The Udacity Nanodegree Program for Machine Learning Engineer with Microsoft Azure has three projects total, the first of which is this one "Optimizing an ML Pipeline in Azure". In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
The information used in this project relates to direct marketing initiatives of a European banking institution. To determine whether a client will sign up for a term is the classification's main objective. The dataset has 32,950 rows with 3,692 positive and 29,258 negative classes, 20 input variables, and 32,950 rows. Age, marital status, debt, loans, and occupations are just a few of the many columns that it has. Both categorical and numerical variables are present. For encoding categorical variables in the dataset, we used a single hot encoding. We decided to use Random Sampling, which supports both discrete and continuous hyperparameters, to sample the hyperparameter space. 

Additionally, it supports our Bandit Policy of early stopping by supporting the termination of low performance runs. Hyperparameter values are chosen at random from the specified search space when using random sampling. In contrast to Bayesian Sampling, which chooses samples based on the results of the previous sample and continuously seeks to improve its results, random sampling randomly selects samples. Bayesian sampling can be resource and computationally demanding. We did not choose Bayesian Sampling because it does not support the early termination policy that we were instructed to use. It should only be used if you have the means to complete it. We decided to terminate the Bandit Policy early as well. 

When the primary metric is outside of the specified slack/factor range of the most successful run, the Bandit Policy terminates the run. The Bandit Policy has the advantage of appearing to be the most configurable when compared to other stopping policies. Additionally, there is no termination policy, which is quite obvious in the ways that it continues. By terminating subpar runs, early termination policies increase computational efficiency. This may be very helpful. Between positive and negative classes, there is a huge imbalance. This might encourage bias in the model.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

For hyperameter tuning, we combine the Sci-KitLearn Logistic Regression algorithm with HyperDrive. The following steps make up the pipeline:
1.Data Gathering
2.Data Cleaning
3.Data Splitting 
4.Hyperparameter Sampling
5.Modeling Activating 
6.Model Testing 
7.Early Stopping Stopping
8.Saving the Model

**What are the benefits of the parameter sampler you chose?**

**What are the benefits of the early stopping policy you chose?**

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

## Future work
Future work might involve enhancing HyperDrive. Instead, you could use Bayesian Parameter Sampling, which samples a probability distribution using Markov Chain Monte Carlo techniques.

AutoML could be improved by increasing the experiment timeout, which would enable more model experimentation. The class imbalance within the datset is another issue that we could address. This would lessen the bias in the models.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
