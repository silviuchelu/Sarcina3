import pickle
import numpy as np

# Încarcă modelul complet (pipeline: vectorizator + clasificator)
with open("product_category_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_top3(text):
    """Returnează top 3 predicții cu probabilități."""
    probs = model.predict_proba([text.lower()])[0]
    classes = model.classes_
    top_idx = np.argsort(probs)[::-1][:3]
    return [(classes[i], probs[i]) for i in top_idx]

if __name__ == "__main__":
    print("Model încărcat ✅. Scrie un titlu de produs (sau 'exit' pentru ieșire).")
    while True:
        s = input("\nTitlu produs: ").strip()
        if s.lower() in ("exit", "quit"):
            break
        top3 = predict_top3(s)
        print("Predicții:")
        for c, p in top3:
            print(f" - {c}: {p:.3f}")
