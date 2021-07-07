from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, SubmitField, validators, ValidationError
import numpy as np

# ラベルから感情名を取得
def getName(label):
    print(label)
    if label == 0:
        return "嬉しい"
    elif label == 1:
        return "悲しい"
    elif label == 2:
        return "怒り"
    else:
        return "Error"

app = Flask(__name__)

class TextForm(Form):
    Content = TextAreaField("分析する文章を入力してください",
        [validators.InputRequired("この項目は入力必須です"), validators.Length(max=140)])

    submit = SubmitField("判定")


@app.route('/', methods = ['GET', 'POST'])
def predicts():
    form = TextForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            return render_template('index.html', form=form)
        else:
            Content = request.form["Content"]

            # ここで分析するコード
            emotionName= getName(np.random.randint(0,3))

            return render_template('result.html', emotionName=emotionName)
    elif request.method == 'GET':
        return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run()