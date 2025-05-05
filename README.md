# Twitter-Sentiment-Analysis.
A machine learning project to predict tweet sentiments (Positive, Negative, Neutral) using an LSTM model built with TensorFlow. Features a Streamlit frontend for real-time sentiment analysis. Trained on Google Colab and deployed using Anaconda/Spyder.
## Features
- Trained an LSTM model on Twitter data to predict sentiment.
- Preprocessed text using regex and tokenization.
- Built a Streamlit app for real-time sentiment prediction.
- Deployed using Anaconda and Spyder.

## Tech Stack
- **Backend**: Python, TensorFlow, Keras, Scikit-learn, Pandas
- **Frontend**: Streamlit
- **Environment**: Google Colab (training), Anaconda/Spyder (deployment)

## Folder Structure
- `backend/twittersentimentanalysis.py`: Model training and preprocessing.
- `frontend/appscript.py`: Streamlit frontend for user input and prediction.

## Setup Instructions
1. Clone the repo: `git clone https://github.com/your-username/Twitter-Sentiment-Analysis.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run frontend/appscript.py`

## Usage
- Enter a tweet-like sentence in the Streamlit app.
- Twitter-Sentiment-Analysis/
├── backend/
│   └── twittersentimentanalysis.py 
├── frontend/
│   └── appscript.py                 
├── README.md
└── requirements.txt
- Click "Predict Sentiment" to see the sentiment (Positive, Negative, Neutral) and confidence scores.

## Future Improvements
- Add real-time Twitter API integration.
- Improve model accuracy with hyperparameter tuning.
- Deploy the app on a cloud platform like Heroku or Streamlit Cloud.
