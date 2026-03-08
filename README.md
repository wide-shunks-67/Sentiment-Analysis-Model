Customer Sentiment Analysis 📊
An end-to-end Deep Learning project that classifies customer reviews into three sentiment categories: Negative, Neutral, and Positive. This repository includes text preprocessing, custom model training using a Bidirectional LSTM, and out-of-the-box inference using a pre-trained Hugging Face Transformer.

🚀 Key Features
Text Preprocessing: Cleans review text by lowercasing and stripping punctuation.

Tokenization & Padding: Uses TensorFlow/Keras Tokenizer and pad_sequences to prepare text data for neural networks.

Custom Deep Learning Model: A robust Bidirectional LSTM model with an Embedding layer and Dropout for regularization.

Pre-trained Transformer Integration: Demonstrates how to load and use a pre-trained state-of-the-art RoBERTa model (cardiffnlp/twitter-roberta-base-sentiment) via Hugging Face's pipeline for comparison.

Inference Pipeline: Includes a ready-to-use prediction function to evaluate new, unseen text instantly.

🛠️ Technologies & Libraries Used
Python 3

TensorFlow / Keras: For building, training, and evaluating the Bidirectional LSTM model.

Scikit-learn: For data splitting (train_test_split).

Hugging Face Transformers: For utilizing pre-trained NLP models.

Pandas & NumPy: For data manipulation and mathematical operations.

Matplotlib: For visualizing training accuracy and loss histories.

Regex (re): For string manipulation and text cleaning.

📂 Dataset
The model is trained on a dataset containing 25,000 customer reviews (Customer_Sentiment.csv).

Classes: The sentiments are mapped to 3 distinct categories:

0 : Negative

1 : Neutral

2 : Positive

Split: 20,000 reviews are used for training and 5,000 for testing.

🧠 Model Architecture
The custom neural network is built using the Keras Sequential API and consists of the following layers:

Embedding Layer: Vocabulary size of 5,000, embedding dimension of 64, and input length of 100.

Bidirectional LSTM: 64 units to capture context from both past and future words.

Dropout Layer: Set to 0.5 to prevent overfitting.

Dense Layer: 32 units with a ReLU activation function.

Output Dense Layer: 3 units with a Softmax activation function to output category probabilities.

⚙️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/wide-shunks-67/Sentiment-Analysis-Model
cd Sentiment-Analysis-Model
Install the required dependencies:

Bash
pip install pandas numpy scikit-learn tensorflow transformers matplotlib
Ensure the dataset (Customer_Sentiment.csv) is placed in the root directory.

Run the Jupyter Notebook:

Bash
jupyter notebook sentiment_analysis.ipynb
💻 Usage / Inference
You can use the built-in predict_sentiment(text) function inside the notebook to classify your own sentences.

Python
# Example Usage
review = "product was great"
print(predict_sentiment(review))
# Output: Positive (Confidence: 0.93)

review2 = "the food delivery was not on time"
print(predict_sentiment(review2))
# Output: Negative (Confidence: 1.00)
📈 Results
The Bidirectional LSTM was trained for 5 epochs using categorical crossentropy and the Adam optimizer.

It achieves incredibly high accuracy on both the training and validation splits. Accuracy and loss progression are plotted via Matplotlib within the notebook.

The notebook also showcases zero-shot predictions using the robust cardiffnlp/twitter-roberta-base-sentiment model to compare confidence scores against the custom LSTM.
