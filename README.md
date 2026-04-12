## Problem Statement
Spam messages are a common issue in digital communication and can be misleading or harmful. Identifying them manually is inefficient, especially when dealing with large volumes of messages.

## Objective
The objective of this project is to build a system that can automatically classify messages as spam or non-spam using Python.

## Features
- Accepts user input from the command line
- Classifies messages instantly
- Uses text vectorization (TF-IDF)
- Simple and easy-to-use interface
- Modular code structure

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Pickle

## Project Structure

```
spam-detector/
│── data/ 
│── model/ 
│── train.py 
│── app.py 
│── requirements.txt 
```

## How It Works
1. The dataset is loaded and preprocessed
2. Text is converted into numerical form using TF-IDF
3. A Naive Bayes model is trained on the data
4. The trained model is saved using pickle
5. User input is analyzed and classified as spam or ham

## Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
Usage
Step 1: Train the Model
python train.py
Step 2: Run the Application
python app.py
Example
Enter a message: Win a free iPhone now
Prediction: spam
