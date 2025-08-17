from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

spark = SparkSession.builder.appName("Decision_Tree").getOrCreate()
start = time.time()

# Load Data
df = spark.read.csv("data.csv", header=True, inferSchema=True)
feature_cols = [
    "Radius_mean","Texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

# Split data
training_data, test_data = df.randomSplit([0.8, 0.2])

# Convert diagnosis column to label (0, 1)
label_indexer = StringIndexer( inputCol="diagnosis", outputCol="label", handleInvalid="skip" )
label_model = label_indexer.fit(training_data)
train_labeled = label_model.transform(training_data)
test_labeled = label_model.transform(test_data)

# Assemble features into single vector in "features" column
assembler = VectorAssembler( inputCols=feature_cols, outputCol="features", handleInvalid="keep" )
train_assembled = assembler.transform(train_labeled)
test_assembled = assembler.transform(test_labeled)

# Train Decision Tree
decision_tree = DecisionTreeClassifier(
    labelCol="label",
    featuresCol="features",
    impurity="gini",
    maxDepth=5,
    minInstancesPerNode=2
)
tree_model = decision_tree.fit(train_assembled)

# Use model to predict the outcome of the training data 
prediction = tree_model.transform(test_assembled).cache()
prediction.select("id", "diagnosis", "label", "prediction", "probability").show(15, truncate=False)

# Calculate and output model stats
accuracy = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="accuracy").evaluate(prediction)
f1 = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="f1").evaluate(prediction)
precision = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="weightedPrecision").evaluate(prediction)
recall = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction",metricName="weightedRecall").evaluate(prediction)

print(f"Accuracy:  {accuracy:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Prediction vs. Reality Distribution
analysis = (prediction.groupBy("label", "prediction").count().orderBy("label", "prediction"))
analysis.show()
print(f"Time Elapsed: {time.time()-start:.2f} seconds")