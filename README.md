# Spam Detector

## Problem Statement
Spam messages are a common issue in digital communication, leading to scams, phishing, and unwanted noise. Identifying them manually is inefficient and error-prone, especially when dealing with large volumes of messages.

## Objective
The objective of this project is to build a reliable machine learning system that automatically classifies SMS messages as either **spam** or **ham** (non-spam) using natural language processing (NLP).

## Technologies Used
- **Python**: Core programming language
- **Pandas**: Data loading, cleaning, and manipulation
- **Scikit-learn**: Machine learning model training, pipelines, and evaluation metrics

## How the Classifier Works
1. **Dataset**: Uses the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) containing 5,169 clean SMS messages (after deduplication).
2. **Preprocessing**: The text is converted into numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency), which measures how important a word is to a message compared to the whole dataset. Stop words are removed to improve accuracy.
3. **Training**: A **Multinomial Naive Bayes** model is trained on the TF-IDF features. This model is highly effective for text classification tasks.
4. **Prediction**: The interactive CLI application accepts user input, transforms it using the saved TF-IDF vectorizer, and predicts the class (Spam or Ham) along with a confidence score.

## Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model
Because the trained model files (`.pkl`) are ignored in version control to save space, you must train the model first. This will clean the dataset, evaluate the model, and generate `model/pipeline.pkl`.

```bash
python train.py
```

### Step 2: Run the Application
Once the model is trained, you can run the interactive command-line interface to test messages.

```bash
python app.py
```

### Example

```text
Enter a message: Congratulations! You have won a free prize. Claim now!

Prediction: SPAM
Confidence: 91.97%
```

## Evaluation Metrics
Based on the testing set (20% of the dataset):
- **Accuracy**: ~97.10%
- **Precision**: 100% (Zero false positives—no ham messages were classified as spam)
- **Recall**: ~78.57% (Identifies most spam messages correctly)
- **F1-Score**: ~0.8800

## Limitations
- The model uses a basic "Bag of Words" (TF-IDF) approach, meaning it does not understand sentence structure or context deeply.
- Typos or creative obfuscation by spammers (e.g., "W!N M0NEY") may bypass the detector.

## Possible Future Improvements
- Implement a more advanced NLP model (e.g., Word2Vec, BERT) if context becomes important.
- Expand the dataset with modern SMS and WhatsApp spam examples.
- Implement hyperparameter tuning (e.g., GridSearchCV) to improve recall.