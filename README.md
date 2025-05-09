# Twitter Suicidal Ideation Detection using NLP and ML/DL Models

![GitHub](https://img.shields.io/badge/python-3.8%2B-blue)
![GitHub](https://img.shields.io/badge/tensorflow-2.x-orange)
![GitHub](https://img.shields.io/badge/flask-2.0-lightgrey)
![GitHub](https://img.shields.io/badge/ktrain-0.3.x-red)

This project aims to develop a system that can detect suicidal ideation on Twitter using Natural Language Processing (NLP) and Machine Learning (ML) models, including Logistic Regression, Support Vector Machines (SVM), Random Forest (RF), Multinomial Naive Bayes (MNB), Ensemble Learning, AdaBoost, Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Convolutional Neural Networks (CNN), and Bidirectional Encoder Representations from Transformers (BERT).
This project aims to identify individuals who may be at risk of suicide and contribute to suicide prevention efforts.
<br> It also includes a **Flask web application** for real-time predictions.

# 📦 Dataset:

- I've used the Twitter Suicide Dataset, which contains Tweets from individuals who have experienced suicidal thoughts and ideations. <br>
- The dataset includes more than 1787  Tweets and comments, and it has been labeled as either indicating suicidal ideation or not.

# ⚙️ Technical Approach

## Data Preprocessing:

- As part of the data preprocessing phase, First I converted Tweets to lower case then cleaned and filtered the text by removing URLs, stop words, and special characters.<br>
- Next, I tokenized the text and performed stemming and lemmatization to reduce the words to their base form.<br>
- This step helped in reducing the dimensionality of the data and improved the efficiency of the models.<br>


## Feature Extraction:

- For feature extraction, I used the Term Frequency-Inverse Document Frequency (TF-IDF) technique.<br>
- Additionally, I also experimented with custom embeddings to capture semantic meaning and reduce the dimensionality of the data.<br>
- These techniques helped in representing the text in a numerical format that could be used by the models to make predictions.<br>


## Machine Learning Models: 

- During the experimentation phase, I trained several ML models including Logistic Regression, Support Vector Machines (SVM), Random Forest (RF), Multinomial Naive Bayes (MNB), Ensemble Learning, and AdaBoost. <br>
- These models were used to make predictions on the test set and identify individuals who may be at risk of suicide.<br>

## Deep Learning Models:

- In addition to the ML models, I also experimented with several DL models such as LSTM, GRU, CNN, and BERT. <br>
- To improve the performance of the BERT model, I fine-tuned it using the ktrain library.<br>
- By fine-tuning the pre-trained BERT model, I was able to capture the context and nuances of the text better, resulting in higher accuracy and F1 scores.<br>
- These DL models were used alongside the ML models to identify individuals who may be at risk of suicide.<br>


# 📈 Results:

- After evaluating the performance of all the models, the best-performing ML model was the ensemble learning model, which achieved an F1 score of 88.67% on the test set.<br>
- On the other hand, best-performing DL models were LSTM-2 Layer and BERT, which achieved 94.18% and 96.00% accuracy, respectively.<br>
- These results demonstrate the effectiveness of using NLP and ML/DL models to detect suicidal ideation on Twitter and highlight the potential for these models to contribute to suicide prevention efforts.

# 🛠️ Web Application Interface


<img src="ui/ui.png" alt="Original Image" width="700">

# 🌟 Conclusion:

My study demonstrates the feasibility of using NLP and ML/DL models to detect suicidal ideation on Twitter.
The models I developed achieved high accuracy and F1 scores, indicating their potential usefulness in identifying individuals who may be at risk of suicide.
These findings suggest that NLP and ML/DL models have the potential to contribute to suicide prevention efforts by identifying individuals who may need help and support.
