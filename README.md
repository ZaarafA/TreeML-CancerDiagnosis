# Tree-Based Machine Learning Models for Cancer Diagonosis
Comparison of Tree Based Machine Learning Models (Decision Tree and Gradient Boosted Tree) using Spark MLlib. 
Data used is from the Breast Cancer Wisconsin Diagnostic dataset.

## To Set Up:
Install pyspark
```
pip install pyspark
```
Run the scripts
```
python3 decision_tree.py
python3 gradient_tree.py
```

### Testing Results
```
Decision Tree                    Gradient Boosted Tree
+-------+------------+-------+   +-------+------------+-------+
| label | prediction | count |   | label | prediction | count |
+-------+------------+-------+   +-------+------------+-------+
| 0.0   | 0.0        | 58    |   | 0.0   | 0.0        | 46    |
| 0.0   | 1.0        | 1     |   | 0.0   | 1.0        | 3     |
| 1.0   | 0.0        | 7     |   | 1.0   | 0.0        | 2     |
| 1.0   | 1.0        | 33    |   | 1.0   | 1.0        | 35    |
+-------+------------+-------+   +-------+------------+-------+
```

### Evaluation

| Metric     | Decision Tree | Gradient Boosted Tree |
|------------|--------------|-----------------------|
| Accuracy   | 0.9192       | 0.9593               |
| F1         | 0.9179       | 0.9590                |
| Precision  | 0.9239       | 0.9599                |
| Recall     | 0.9192       | 0.9593                |

### Comparison and Analysis
Both models performed well at classifying tumors based on the features, but the Gradient Boosted Tree consistently outperformed the Decision Tree across all metrics. The Gradient Boosted model correctly classified more tumors and based on the accuracy and precision, had fewer false positives overall. Based on the recall, it was able to identify more actual malignant cases, reducing false negatives, which are more dangerous in application.

The decision tree was more conservative when it came to diagnosing a tumor as malignant. The decision tree had substantially more false negatives but had fewer false positives than the Gradient Boosted Tree. This is a good thing for the GBT because in practice, a false negative could have more dangerous repercussions, versus a false positive, which may just lead to an unnecessary doctor's visit. 

### Limitations and Suggested Improvements
- Small Dataset: The dataset only had 569 rows. This makes it harder for the models to generalize because it doesn’t have access to a wide range of variation. The solution would be to train the model on a much larger dataset.
- Hyperparameters: The hyperparameters for both models were chosen manually. I chose them by adjusting the values and seeing if they improved. Using an automatic technique like Grid Search or Random Search could find the best parameter values. 

