import pickle
import os

if not os.path.exists("model/model.pkl") or not os.path.exists("model/vectorizer.pkl"):
    print("Model not found. Please run train.py first.")
    exit()

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

msg = input("Enter a message: ")

msg_vector = vectorizer.transform([msg])

prediction = model.predict(msg_vector)

print("Prediction:", prediction[0])