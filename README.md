SMS Spam Detection is a machine learning project that classifies SMS messages as spam or ham (non-spam) using a neural network built with TensorFlow and Keras. The model is trained on the SMS Spam Collection Dataset, which contains over 5,000 labeled SMS messages.

📩 SMS Spam Detection using TensorFlow This project detects whether an SMS message is spam or ham (not spam) using a deep learning model built with TensorFlow and Keras.

🚀 Features Preprocessing and tokenizing text data
Building a neural network model using TensorFlow
Training and validating the model
Saving/loading model and tokenizer
Predicting new messages (via main.py)
Visualizing accuracy and loss over training epochs

🧠 Model Architecture
Embedding Layer: 5000 vocab size, 16 dimensions
GlobalAveragePooling1D
Dense Layer: 24 units, ReLU
Output Layer: 1 unit, Sigmoid

🗂️ Project Structure
sms-spam-detection/
│
├── spam.csv # Dataset (SMS spam collection)
├── train_model.py # Script to preprocess data and train model
├── main.py # Script to load model & tokenizer, predict SMS text
├── spam_model.h5 # Saved trained model
├── tokenizer.pkl # Saved tokenizer for text preprocessing
├── requirements.txt # All required dependencies
└── README.md # Project documentation

📦 Requirements
Python 3.10
TensorFlow 2.14.0
NumPy 1.26.4
scikit-learn
pandas
matplotlib

✍️ Author
Rohan Pimparkar
