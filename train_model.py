# ==========================================
# train_model.py
# Sarcina 3 - Predicția categoriei produsului pe baza titlului
# ==========================================
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Citire CSV
df = pd.read_csv("IMLP4_TASK_03-products.csv")
df.columns = [c.strip() for c in df.columns]
df = df[['Product Title', 'Category Label']].dropna()
df = df.rename(columns={'Product Title': 'title', 'Category Label': 'category'})

# 2. Curățare text
def clean_text(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['title'] = df['title'].apply(clean_text)
df = df[df['title'] != ""]

# 3. Filtrare etichete rare
min_count = 50
valid = df['category'].value_counts()
keep = valid[valid >= min_count].index
df = df[df['category'].isin(keep)]

# 4. Împărțire train/test
X = df['title'].values
y = df['category'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# 5. Model TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=30000)),
    ('clf', LogisticRegression(max_iter=1000, solver='saga',
                               multi_class='multinomial', n_jobs=-1))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 6. Evaluare
acc = accuracy_score(y_test, y_pred)
print(f"✅ Acuratețe pe setul de test: {acc:.3f}")
print("\nRaport de clasificare:\n", classification_report(y_test, y_pred))

# 7. Matrice de confuzie
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
sns.heatmap(cm, xticklabels=pipeline.classes_, yticklabels=pipeline.classes_,
            annot=False, cmap='Blues')
plt.title('Matrice de confuzie')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.close()

# 8. Salvare model
with open("product_category_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("💾 Modelul a fost salvat în product_category_model.pkl")
