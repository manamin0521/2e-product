from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, SubmitField, validators, ValidationError

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

            return render_template('result.html')
    elif request.method == 'GET':
        return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run()