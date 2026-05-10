import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

x = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

x_vectorized = vectorizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))