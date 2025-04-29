import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("AI_Resume_Screening.csv")

df['Skills'] = df['Skills'].fillna('')  
df['Education'] = df['Education'].fillna('')  
df['Certifications'] = df['Certifications'].fillna('')  

df['Combined_Feature'] = df['Skills'] + " " + df['Education'] + " " + df['Certifications']

X_train, X_test, y_train, y_test = train_test_split(df['Combined_Feature'], df['Job_Role'], test_size=0.2, random_state=42)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

joblib.dump(model, "job_role_model.pkl")
joblib.dump(model.named_steps["tfidf"], "vectorizer.pkl")

print("Model and vectorizer saved successfully!")
