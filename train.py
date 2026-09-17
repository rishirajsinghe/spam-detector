import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def main():
    data_path = "data/spam.csv"
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # Check required columns
    if 'label' not in df.columns or 'message' not in df.columns:
        print("Error: Dataset must contain 'label' and 'message' columns.")
        return

    initial_count = len(df)
    
    # Remove empty messages
    df = df.dropna(subset=['message'])
    df = df[df['message'].str.strip() != '']
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['message'])
    
    # Normalize labels
    df['label'] = df['label'].str.lower().str.strip()
    # Keep only valid labels if any noise exists
    df = df[df['label'].isin(['spam', 'ham'])]

    print(f"Data cleaned. Removed {initial_count - len(df)} empty/duplicate/invalid rows.")
    
    # Report numbers
    print(f"Total messages: {len(df)}")
    print(f"Ham messages: {len(df[df['label'] == 'ham'])}")
    print(f"Spam messages: {len(df[df['label'] == 'spam'])}")
    
    # Prepare data
    X = df['message']
    y = df['label']
    
    # Split data (using fixed random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining the model...")
    # Create Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', MultinomialNB())
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    # Note: pos_label='spam' to calculate metrics for spam detection
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam')
    rec = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    
    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(f"[[TN={cm[0][0]} FP={cm[0][1]}]")
    print(f" [FN={cm[1][0]} TP={cm[1][1]}]]")
    print("--------------------------\n")
    
    # Save model
    if not os.path.exists("model"):
        os.makedirs("model")
        
    # We save the entire pipeline so it includes both vectorizer and model
    model_path = "model/pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
        
    print(f"Model saved successfully to {model_path}!")

if __name__ == "__main__":
    main()