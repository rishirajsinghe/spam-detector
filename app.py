import os
import pickle

def main():
    model_path = "model/pipeline.pkl"
    
    if not os.path.exists(model_path):
        print("Error: Trained model not found.")
        print("Please run 'python train.py' first to train and save the model.")
        return

    print("Loading model...")
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    print("-" * 40)
    print("Spam Detector is ready!")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 40)
    
    while True:
        try:
            msg = input("\nEnter a message: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if not msg:
            continue
            
        if msg.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
            
        # Predict
        prediction = pipeline.predict([msg])[0]
        probabilities = pipeline.predict_proba([msg])[0]
        
        # Display results clearly
        print(f"\nPrediction: {prediction.upper()}")
        
        # Display confidence (optional but good for a demo)
        classes = pipeline.classes_
        # Ensure we're printing the correct probability for the prediction
        pred_idx = list(classes).index(prediction)
        confidence = probabilities[pred_idx] * 100
        print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()