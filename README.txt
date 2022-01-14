Fake News Classifier

Description:
This project uses a Linear Support Vector Machine algorithm to classify Twitter
Text input as 'Real' or 'Fake' (training/test set from the repository). The algorithm
achieves an F1 'Micro' accuracy of 87%.

Installation guide:
Python version: 3.9
Libraries needed: pandas, nltk, sklearn, csv, re
Notes: 
1. The code includes the download statements for online libraries, however
there may be an inconsistency on different machines(more nltk.download(..) may be needed)
2. The training and test data sets are inside the .zip archive(since the filename is hard-coded,
the files need to be inside the same folder).

Run from command prompt with the path to file: "py .\fake_news_classifier.py"


