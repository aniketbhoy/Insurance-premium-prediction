import pickle
import pandas as pd
from pathlib import Path

# with open('model.pkl','rb') as f:
#     model=pickle.load(f)
BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "model.pkl", "rb") as f:
    model = pickle.load(f)

MODEL_VERSION='1.0.0'

class_labels=model.classes_.tolist()
def predict_output(user_input: dict):
    df=pd.DataFrame([user_input])
    predicted_class=model.predict(df)[0]
    prob=model.predict_proba(df)[0]
    confidence=max(prob)
    class_probs=dict(zip(class_labels,map(lambda p: round(p,4), prob)))
    return {
        'predicted_category':predicted_class,
        'confidence':round(confidence,4),
        'class_probabilities':class_probs
    }