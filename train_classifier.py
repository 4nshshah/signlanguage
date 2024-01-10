import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open("./data.pickle", "rb"))

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"], dtype=np.float32)
print(data)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
f1 = sklearn.metrics.f1_score(y_test, y_predict, average="weighted")
precision = sklearn.metrics.precision_score(y_test, y_predict, average="weighted")
recall = sklearn.metrics.recall_score(y_test, y_predict, average="weighted")
print("F1 score: {}".format(f1))
print("Precision score: {}".format(precision))
print("Recall score: {}".format(recall))


print("{}% of samples were classified correctly !".format(score * 100))

f = open("model.p", "wb")
pickle.dump({"model": model}, f)
f.close()
