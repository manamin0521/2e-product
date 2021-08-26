# predictに必要なもの
# 学習済みmodel
# maxlenの値
# 保存済みtokenizerインスタンス

from keras.preprocessing.sequence import pad_sequences
from keras import models
import numpy as np
# import re
import pickle
# import json

from flask import Flask, render_template, request
from werkzeug.utils import redirect
from wtforms import Form, TextAreaField, SubmitField, validators, ValidationError
from janome.tokenizer import Tokenizer
# import numpy as np
# from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN

app = Flask(__name__)


class TextForm(Form):
    Content = TextAreaField("分析したい文章を入力して下さい",
                            [validators.InputRequired("この項目は入力必須です"), validators.Length(max=140)])

    submit = SubmitField("診断する")


@app.route('/', methods=['GET'])
def top():
    return render_template('top.html')

@app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/index.html', methods=['GET', 'POST'])
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
            emotions = sorted(emotions.items(),
                              key=lambda x: x[1], reverse=True)
            emotions_label_dic = {'Avg. Readers_Surprise': '驚き😲', 'Avg. Readers_Sadness': '悲しみ😭', 'Avg. Readers_Joy': '喜び😄', 'Avg. Readers_Anger': '怒り💢',
                                  'Avg. Readers_Fear': '恐れ😨', 'Avg. Readers_Disgust': '嫌悪😠', 'Avg. Readers_Trust': '信頼🤝', 'Avg. Readers_Anticipation': '期待😆'}

            return render_template('result.html', emotions=emotions, emotions_label_dic=emotions_label_dic)
    elif request.method == 'GET':
        return render_template('index.html', form=form)


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


t = Tokenizer()


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


def each_predict(text, model_name):

    wakati_ed = wakati(text)
    wakati_list = []
    wakati_list.append(wakati_ed)

    loaded_tokenizer = load_text_tokenizer('tokenizer_ja')
    tokenized = loaded_tokenizer.texts_to_sequences(wakati_list)

    np_arr = np.array(tokenized)
    np_arr = pad_sequences(np_arr, maxlen=124)

    loaded_model = models.load_model(model_name)
    result = loaded_model.predict(np_arr)
    # print(result)
    # if result >= 0.5:
    #   print(model_name)
    return result
# 日本語8つの感情


def predict(text):
    result_dic = {}
    model_list = ['Avg. Readers_Surprise', 'Avg. Readers_Sadness', 'Avg. Readers_Joy', 'Avg. Readers_Anger',
                  'Avg. Readers_Fear', 'Avg. Readers_Disgust', 'Avg. Readers_Trust', 'Avg. Readers_Anticipation']
    for model in model_list:
        result_dic[model] = each_predict(text, model + '.h5')
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
