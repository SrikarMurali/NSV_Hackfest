from flask import Flask, jsonify, request
from keras.models import load_model
import gensim
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

app = Flask(__name__)
app.config["DEBUG"] = True

file_name = 'model.h5'

model = load_model('model.h5')
wv_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)  

def clean_lemmatize(doc):
    """lemmatize and clean doc"""
    # Build Lemmatizer
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

# Run word_list into an input volume
def clean(phrase_list):
    lemmatized_list = []
    for phrase in phrase_list:
        #assert type(word) == list
        if type(phrase) == list:
            temp = [clean_lemmatize(x) for x in phrase]
            lemmatized_list.append(temp)
    for i in range(len(lemmatized_list)):
        temp = [x for x in lemmatized_list[i] if x!= '' and x.isalpha()]
        lemmatized_list[i] = temp
    return lemmatized_list

def input_volume(phrase_list, timesteps):
    clean(phrase_list)
    w2v = np.zeros([len(phrase_list), timesteps, 300])
    c1 = 0
    for word in phrase_list:
        x = np.zeros([6, 300])
        if type(word) == list:
            c2 = 0
            for w in word:
                try: 
                    x[c2] = wv_model[w]
                    c2 += 1
                except KeyError:
                    pass
        elif type(word) == str:
            x[0] = wv_model[word]
        w2v[c1] = x
        c1 += 1
    return w2v

def predict_behavior(phrase):
    phrase_list = [phrase.split(' ')]
    input_vol = input_volume(phrase_list, 6)
    y_pred = model.predict(input_vol)
    lexical_predictions = []
    for label in y_pred:
        for i, x in enumerate(label):
            if x > 0.8: 
                lexical_predictions.append(types_of_violence[i])
    return lexical_predictions

@app.route("/")
def index():
    return "Hello World!"

@app.route('/api/violence', methods=['GET'])
def api_get_violence_types():
    if 'value' in request.args:
        value = request.args['value']
        predicted_types_violence = predict_behavior(value)
        return jsonify(predicted_types_violence)   
    else:
        return "Error: No behavior field provided. Please specify a behavior."
    
    return "Error"

if __name__ == "__main__":
    app.run(host='0.0.0.0')