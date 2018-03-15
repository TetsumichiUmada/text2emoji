# Emoji Prediction

While texting to your friends, can you tell their emotion? Are they happy? Nowadays, people often send text messages to each other. However, it's difficult for people to understand and know a sender's emotions based on text messages, especially from not close friends. This project tries to build a classifier to predict sentiment associated with a text and represent it as an emoji.

## Quickstart
To use this project, it's required to install python3, jupyter notebook, and some python libraries.

### Install
#### Install python3
If you don't have python3 on your computer, there are two options:
+ Download python3 from [Anaconda](https://www.anaconda.com/download/), which includes Python, Jupyter Notebook, and the other libraries.
+ Download python3 from [python.org](https://www.python.org/downloads/)

#### Install packages
All packages used for this project are written in `requirements.txt`. To install, you can run
```
$ pip3 install -r requirements.txt
```

#### Download project
To download this project repository, you can run
```
$ git clone https://github.com/TetsumichiUmada/text2emoji.git
```

#### Run jupyter notebook
To start jupyter notebook, you move to the directory with `cd/path_to/text2emoji`, then run
```
$ jupyter notebook
```
See [Running the Notebook](https://jupyter.readthedocs.io/en/latest/running.html#running) for more details.

## Project Details
The goal of this project is to predict an emoji that is associated with a text message. To accomplish this task, we train and test several supervised machine learning models on a data to predict a sentiment associated with a text message. Then, we represent the predicted sentiment as an emoji.

### Data Sets
The data comes from the [DeepEmoji/data](https://github.com/bfelbo/DeepMoji/tree/master/data) repository. Since the file format is a pickle, we wrote a python 2 script to covert a pickle to a txt file. The data (both pickle and txt files) and scripts are stored in the text2emoji/data directory.

Among the available data on the repository, we use the PsychExp dataset for this project. In the file, there are 7840 samples, and each line contains a text message and its sentimental labels which are represented as a vector `[joy, fear, anger, sadness, disgust, shame, guilt]`.

In the txt file, each line is formatted like below:

```
[ 1.  0.  0.  0.  0.  0.  0.] Passed the last exam.
```

Since the first position of the vector is 1, the text is labeled as a joy.

For more information about the original data sets, please check [DeepEmoji/data](https://github.com/bfelbo/DeepMoji/tree/master/data) and text2emoji/data.


### Preprocess and Features
To be able to build a model, we need to make features. From the text, we made [n-grams](https://en.wikipedia.org/wiki/N-gram) (a contiguous sequence of n words from a given text) with a range from 1 to 4. The feature also includes punctuations such as !, ?, or .. All texts are converted into lower cases.


### Models and Results
To choose the best model for this project, we tried four different classifiers. To train and test each model, the data was randomized. Then it was split into 80-20 ratio where 80% for the training and 20% for the testing. We kept 20% to make sure that we had enough amount of the data for testing.

We evaluate the performance of each model by calculating an accuracy score. The accuracy score is simply the proportion of classifications that were done correctly and is calculated by

$$
\text{Accuracy} = \frac{\text{number of correct classifiers}}{\text{total number of classifications made}}
$$

For this project, we tried following classifiers and their accuracy scores are summarized in the table below.

| Classifier                | Training Accuracy | Test Accuracy |
| ------------------------- | ----------------- | ------------- |
| [SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)                       |         0.1458890 |     0.1410428 |
| [LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)                 |         0.9988302 |     0.5768717 |
| [RandomForestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)    |         0.9911430 |     0.4304813 |
| [DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)    |         0.9988302 |     0.4585561 |

Based on the accuracy scores, it seems like SVC (support vector machine) classier is working fine, but the accuracy is low. The LinearSVC is working good although the model is overfitting (meaning that the training accuracy is higher and test accuracy is low). We can observe the same phenomena for the other classifiers. By choosing the classifier with the highest test accuracy score, the LinearSVC seems to be good.

### Error Analysis

We analyzed the classification results from the LinearSVC model, using the confusion matrix. The rows represent the true labels and the columns are predicted labels.

![](images/confusion_matrix.png)

Based on the above table, the classifier tends to misclassify text messages with guilt, shame, and anger. This happens probably because the text does not have any word which characterizes its sentiment. On the other hand, because joy message more likely to have words such as good, like, and happy, the classifier is able to well find out if the given message is happy or not.

### Future Work
To be able to accurately analyze the text, we probably need to have more data to train the classifiers. At the same time, we can more experiment with engineering features. It might work if we use a Chi-squared test to find out more informative tokens. We might also be possible to build a deep learning for the sentimental classification.

### Demo (outputs from the classification model)

😂 Thank you for dinner!       
😢 I don't like it          
😱 My car skidded on the wet street        
😢 My cat died       


### Reference
+ [DeepMoji](https://www.media.mit.edu/projects/deepmoji/overview/)
+ [DeepMoji GitHub](https://github.com/bfelbo/DeepMoji)
+ [Multiclass and multilabel algorithms](http://scikit-learn.org/stable/modules/multiclass.html)
+ [sklearn.svm.SVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
+ [sklearn.svm.LinearSVC](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)
+ [sklearn.ensemble.RandomForestClassifie](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
+ [sklearn.tree.DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
