from flask import Flask, request, render_template
from utils import tokenize
import pickle
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, static_url_path='/data', template_folder='web')

# Load models
model_toxic = joblib.load('./data/model_toxic.pkl')
model_severe_toxic = joblib.load('./data/model_severe_toxic.pkl')
model_identity_hate = joblib.load('./data/model_identity_hate.pkl')
model_insult = joblib.load('./data/model_insult.pkl')
model_obscene = joblib.load('./data/model_obscene.pkl')
model_threat = joblib.load('./data/model_threat.pkl')
tfidf = joblib.load('./data/tfidf.pkl')

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/predict')
def load_predict():
    return render_template('predict.html')

@app.route('/dataset')
def load_dataset():
    return render_template('dataset.html')

@app.route('/predict', methods=['POST'])
def post_predict():
    """
        Get the post typed then apply TFIDF vectorizer and predict using trained models
    """
    text = request.form['text']

    comment = tfidf.transform([text])

    dict_preds = {}

    dict_preds['pred_toxic'] = model_toxic.predict_proba(comment)[:, 1][0]
    dict_preds['pred_severe_toxic'] = model_severe_toxic.predict_proba(comment)[:, 1][0]
    dict_preds['pred_identity_hate'] = model_identity_hate.predict_proba(comment)[:, 1][0]
    dict_preds['pred_insult'] = model_insult.predict_proba(comment)[:, 1][0]
    dict_preds['pred_obscene'] = model_obscene.predict_proba(comment)[:, 1][0]
    dict_preds['pred_threat'] = model_threat.predict_proba(comment)[:, 1][0]

    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = "{0:.2f}%".format(perc)

    return render_template('predict.html', text=text,
                           pred_toxic=dict_preds['pred_toxic'],
                           pred_severe_toxic=dict_preds['pred_severe_toxic'],
                           pred_identity_hate=dict_preds['pred_identity_hate'],
                           pred_insult=dict_preds['pred_insult'],
                           pred_obscene=dict_preds['pred_obscene'],
                           pred_threat=dict_preds['pred_threat'])


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8082)