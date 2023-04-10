# Project 1 - Optimizing an ML Pipeline in Azure

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
RandomParameterSampling, which draws hyperparameters at random from a predetermined search space, is the parameter sampler of choice in the code.
The ease of use and lack of prerequisite information or presumptions regarding the search space are two advantages of using RandomParameterSampling. By randomly selecting from a wide range of values for each hyperparameter, it also enables a more thorough exploration of the search space.
As it does not necessitate a thorough search of all possible combinations, RandomParameterSampling can be computationally efficient for high-dimensional search spaces. Instead, it chooses hyperparameter values at random, which enables the search to concentrate on interesting regions of the search space.

Overall, RandomParameterSampling can be computationally effective for high-dimensional search spaces and is a good option when there is no prior knowledge of the search space or when a thorough search of the space is desired.

**What are the benefits of the early stopping policy you chose?**

BanditPolicy, a kind of adaptive early termination policy, is the early stopping policy selected in the code. Inefficient runs are terminated by the policy earlier than the maximum number of specified iterations, conserving computational resources and enabling quicker experimentation.

By stopping early in the experiment runs that are unlikely to produce promising results, BanditPolicy can reduce costs, which is one of its advantages. This is so that, based on the best performing run and a slack factor, the policy can evaluate the performance of the runs as they proceed and stop those that are most likely to produce positive results. As fewer runs are required to produce a good result, this can save time and resources.

By stopping runs that are likely to perform poorly, BanditPolicy can also increase the effectiveness of hyperparameter search by freeing up resources to be allocated to runs with a higher chance of success. This can improve the likelihood of discovering a good set of hyperparameters and produce better results faster.

In general, BanditPolicy is a good option when the search space is large and it is desired to minimize the quantity of runs required to identify the best hyperparameters. Saving computational resources and cutting costs are additional benefits.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**
By automatically examining various algorithms and hyperparameters, AutoML Azure generates the best machine learning model and hyperparameters. The user-selected performance metric, such as accuracy or AUC, determines the model and hyperparameters, and the search is carried out using time- and resource-efficient methods like Bayesian optimization and ensemble modeling.

<img width="1039" alt="image" src="https://user-images.githubusercontent.com/110788191/230802102-84f2e7b5-dd0e-4c10-9da3-eca225e3ebbc.png">

However, the best algorithm ultimately turned out to be the MaxAbsScaler, LightGBM, with an accuracy of 0.91. Also, other models trained and evaluated are listed. Here the following code snippets from AutoML model explainbility. 
```json
{
    "spec_class": "preproc",
    "class_name": "MaxAbsScaler",
    "module": "sklearn.preprocessing",
    "param_args": [],
    "param_kwargs": {},
    "prepared_kwargs": {}
}
```
The following code snippet shows the hyperparameter setting configured for model.

```json
{
    "spec_class": "sklearn",
    "class_name": "LightGBMClassifier",
    "module": "automl.client.core.common.model_wrappers",
    "param_args": [],
    "param_kwargs": {
        "min_data_in_leaf": 20
    },
    "prepared_kwargs": {}
}
```
## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

When compared to the HyperDrive Model, the model created by AutoML had a marginally superior accuracy, as can be seen in the final Jupyter Notebook. MaxAbsScaler, LightGBM, one of AutoML's finest models, had an accuracy of 0.9153, and the HyperDrive Model had an accuracy of 0.9060. The HyperDrive architecture was limited to Sci-KitLearn's Logistic Regression. Around 20 different models might be evaluated by the AutoML, which has access to a large range of models. In comparison to AutoML, HyperDrive is undoubtedly at a disadvantage given that AutoML offers more than 20 models from which to choose during an experiment.

In order to find the appropriate model architecture, hyperparameters, and features depending on a given dataset and evaluation metric, AutoML employs algorithms. Hyperdrive is a tool for adjusting hyperparameters that employs an optimization method to identify the ideal collection of hyperparameters for a predetermined model architecture.
Both AutoML and hyperdrive are used to optimize machine learning models; however, AutoML is more versatile and can also look for the best model architecture, whilst hyperdrive exclusively focuses on hyperparameter tweaking.

## Future work
Future work might involve enhancing HyperDrive. Instead, you could use Bayesian Parameter Sampling, which samples a probability distribution using Markov Chain Monte Carlo techniques.

AutoML could be improved by increasing the experiment timeout, which would enable more model experimentation. The class imbalance within the datset is another issue that we could address. This would lessen the bias in the models.

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
