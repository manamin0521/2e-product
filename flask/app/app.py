# predictに必要なもの
# 学習済みmodel
# maxlenの値
# 保存済みtokenizerインスタンス

from typing import AsyncGenerator
from warnings import filterwarnings
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras.saving.save import load_model
import numpy as np
# import re
import pickle
# import json

from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, SubmitField, validators, ValidationError
from janome.tokenizer import Tokenizer
# import numpy as np
# from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

import os
from flask import url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.model_selection import train_test_split
t = Tokenizer()

base_dir = os.path.dirname(__file__)

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(base_dir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Data(db.Model):
    __tablename__ = 'data'

    id = db.Column(db.Integer, primary_key=True)
    review = db.Column(db.Text)
    surprise = db.Column(db.Integer)
    sadness = db.Column(db.Integer)
    joy = db.Column(db.Integer)
    anger = db.Column(db.Integer)
    fear = db.Column(db.Integer)
    disgust = db.Column(db.Integer)
    trust = db.Column(db.Integer)
    anticipation = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime)

    def __init__(self, review, surprise, sadness, joy, anger, fear, disgust, trust, anticipation):
        self.review = review
        self.surprise = surprise
        self.sadness = sadness
        self.joy = joy
        self.anger = anger
        self.fear = fear
        self.disgust = disgust
        self.trust = trust
        self.anticipation = anticipation
        self.timestamp = datetime.now()
        # formatted = datetime.datetime.strptime(test, "%Y-%m-%d %H:%M:%S.%f")

class TextForm(Form):
    Content = TextAreaField("分析したい文章を入力してね",
        [validators.InputRequired("この項目は入力必須です"), validators.Length(max=140)])

    submit = SubmitField("診断する")

def get_dataframe():
    conn = db.session.bind
    sql = "SELECT * FROM data"
    df = pd.read_sql(sql, conn)
    df = df.dropna()
    return df

def get_train_data(emotion):
    df = get_dataframe()
    df['review'] = df['review'].apply(wakati)
    X_train, X_test, y_train, y_test = train_test_split(df[['review']], df[['{}'.format(emotion)]], test_size=0.5, random_state=0)
    # X_train = df[['review']]
    # y_train = df[['joy']]
    loaded_tokenizer = load_text_tokenizer('tokenizer_ja')
    # loaded_tokenizer.fit_on_texts(X_train['review'])
    x_train = loaded_tokenizer.texts_to_sequences(X_train['review'])
    x_test = loaded_tokenizer.texts_to_sequences(X_test['review'])
    x_train_np = np.array(x_train)
    x_test_np = np.array(x_test)
    x_train1 = pad_sequences(x_train_np, maxlen=124)
    x_test1 = pad_sequences(x_test_np, maxlen=124)
    y_train_np = y_train.values
    y_test_np = y_test.values
    return x_train1, x_test1, y_train_np, y_test_np
emotion_dict = {
  'Avg. Readers_Surprise':'surprise',
  'Avg. Readers_Sadness':'sadness',
  'Avg. Readers_Joy':'joy',
  'Avg. Readers_Anger':'anger',
  'Avg. Readers_Fear':'fear',
  'Avg. Readers_Disgust':'disgust',
  'Avg. Readers_Trust':'trust',
  'Avg. Readers_Anticipation':'anticipation'
}
def update_model():
    model_list = ['Avg. Readers_Surprise', 'Avg. Readers_Sadness', 'Avg. Readers_Joy', 'Avg. Readers_Anger', 'Avg. Readers_Fear', 'Avg. Readers_Disgust', 'Avg. Readers_Trust', 'Avg. Readers_Anticipation']
    df = get_dataframe()
    timestamp = df.iloc[-1, -1]
    timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f")
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S.%f')
    os.mkdir('model/{}'.format(timestamp))
    for model_name in model_list:
        x_train, x_test, y_train, y_test = get_train_data(emotion_dict[model_name])
        loaded_model = models.load_model(model_name + '.h5')
        loaded_model.fit(x_train, y_train, batch_size=1, epochs=3, validation_data=(x_test, y_test))
        loaded_model.save('model/{}/{}'.format(timestamp, model_name + '.h5'))

def get_latest_model():
    models_labels = os.listdir('model')
    models_created_at = []
    for model_label in models_labels:
        timestamp = datetime.strptime(model_label, "%Y%m%d-%H%M%S.%f")
        models_created_at.append(timestamp)
    latest_models = max(models_created_at)
    latest_models = latest_models.strftime('%Y%m%d-%H%M%S.%f')
    return latest_models

# from app import Data, db, get_train_data, text2vec, each_predict
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = get_train_data()
# from keras import models
# model_name = 'Avg. Readers_Anger.h5'
# loaded_model = models.load_model(model_name)
# text = '今日も元気に頑張るよ'
# np_arr = text2vec(text)
# loaded_model.predict(np_arr)
# loaded_model.fit(x_train, y_train, batch_size=1, epochs=3, validation_data=(x_test, y_test))
# loaded_model.predict(np_arr)

def text2vec(text):

  wakati_ed = wakati(text)
  wakati_list = []
  wakati_list.append(wakati_ed)

  loaded_tokenizer = load_text_tokenizer('tokenizer_ja')
  tokenized = loaded_tokenizer.texts_to_sequences(wakati_list)

  np_arr = np.array(tokenized)
  np_arr = pad_sequences(np_arr, maxlen=124)

  return np_arr







@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form = TextForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            return render_template('index.html', form=form)
        else:
            Content = request.form["Content"]

            # ここで分析するコード
            # emotions = predict(Content)

            tmp = predict(Content)
            emotions = {}
            for key, value in tmp.items():
              emotions[key] = round(value[0][0]*100, 1)
            emotions = sorted(emotions.items(), key=lambda x:x[1], reverse=True)
            emotions_label_dic = {'Avg. Readers_Surprise':'驚き😲', 'Avg. Readers_Sadness':'悲しみ😭', 'Avg. Readers_Joy':'喜び😄', 'Avg. Readers_Anger':'怒り💢', 'Avg. Readers_Fear':'恐れ😨', 'Avg. Readers_Disgust':'嫌悪😠', 'Avg. Readers_Trust':'信頼🤝', 'Avg. Readers_Anticipation':'期待😆'}

            return render_template('result.html', emotions=emotions, emotions_label_dic=emotions_label_dic, text=Content)
    elif request.method == 'GET':
        return render_template('index.html', form=form)

@app.route("/save_data", methods=['POST'])
def save_data():
  if request.method == 'POST':
    review = request.form.get('Content')
    surprise = request.form.get('surprise')
    sadness = request.form.get('sadness')
    joy = request.form.get('joy')
    anger = request.form.get('anger')
    fear = request.form.get('fear')
    disgust = request.form.get('disgust')
    trust = request.form.get('trust')
    anticipation = request.form.get('anticipation')

    data = Data(review, surprise, sadness, joy, anger, fear, disgust, trust, anticipation)
    db.session.add(data)
    db.session.commit()
    df = get_dataframe()
    # if df.iloc[-1, 0] % 10 == 0:
    update_model()
  return render_template('thanks.html')



# def remove_punct(text):
#     """
#     顔文字（emoticons）のみを残し､句読文字（punctation）の削除を行う｡
#     """
#     # 目の部分: : or ; or =
#     # 鼻の部分: -
#     # 口の部分: ) or ( or D or P
#     pattern = re.compile(r"(?::|;|=)(?:-)?(?:\)|\(|D|P)")
#     # パターンに当てはまる文字列をすべて抽出して､リストに格納
#     # いったんリストに保存しておいて､あとですべての記号を削除したtextに付け足す
#     emoticons = pattern.findall(text)
    # # 文頭などにある大文字を小文字に変換
    # lower = text.lower()
    # # [\W]+で記号の並びを捕捉して空白ひとつに置き換える
    # removed = re.sub(r"[\W]+", " ", lower)
    # # 顔文字を半角スペース区切りでひとつの文字列に結合
    # emoticons = " ".join(emoticons)
    # # 顔文字に含まれる鼻の部分を削除する
    # # 鼻があってもなくても､目と口が同じなら同じ顔文字として認識されるようになる
    # emoticons = emoticons.replace("-","")
    # # lowerとemoticonsを結合
    # # 単語を区切るため､間には半角スペースを入れておく
    # connected = removed + ' ' + emoticons
    # return connected

def load_text_tokenizer(file_name):
  # loading
  with open(file_name+".pickle", 'rb') as handle:
      return pickle.load(handle)

# max_len = 124

# from janome.tokenizer import Tokenizer
# t = Tokenizer()
def wakati(text):
  tokens = t.tokenize(text)
  tmp_sentence = ''
  for token in tokens:
    tmp_sentence += token.surface + ' '
  return tmp_sentence

def load_text_tokenizer(file_name):
  # loading
  with open(file_name+".pickle", 'rb') as handle:
      return pickle.load(handle)

def each_predict(text, model_name, dir_name):

  wakati_ed = wakati(text)
  wakati_list = []
  wakati_list.append(wakati_ed)

  loaded_tokenizer = load_text_tokenizer('tokenizer_ja')
  tokenized = loaded_tokenizer.texts_to_sequences(wakati_list)

  np_arr = np.array(tokenized)
  np_arr = pad_sequences(np_arr, maxlen=124)
  path = 'model/{}/{}'.format(dir_name, model_name)
  loaded_model = models.load_model(path)
  result = loaded_model.predict(np_arr)
  # print(result)
  # if result >= 0.5:
  #   print(model_name)
  return result
# 日本語8つの感情
def predict(text):
  dirname = get_latest_model()
  result_dic = {}
  model_list = ['Avg. Readers_Surprise', 'Avg. Readers_Sadness', 'Avg. Readers_Joy', 'Avg. Readers_Anger', 'Avg. Readers_Fear', 'Avg. Readers_Disgust', 'Avg. Readers_Trust', 'Avg. Readers_Anticipation']
  for model in model_list:
    result_dic[model] = each_predict(text, model + '.h5', dirname)
  return result_dic



# def predict(text):

#   removed = remove_punct(text)
#   removed_list = []
#   removed_list.append(removed)

#   loaded_tokenizer = load_text_tokenizer('tokenizer')
#   tokenized = loaded_tokenizer.texts_to_sequences(removed_list)

#   np_arr = np.array(tokenized)
#   max_len = 2505 #自動化したい
#   np_arr = pad_sequences(np_arr, maxlen=max_len)

#   loaded_model = models.load_model('ml14_lstm.h5')
#   result = loaded_model.predict(np_arr)
#   if result >= 0.5:
#     return 'positive'
#   else:
#     return 'negative'

if __name__ == "__main__":
    app.run(debug=True)

