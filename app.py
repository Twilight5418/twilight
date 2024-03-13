import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = joblib.load('gradient_boosting_model.pkl')

@app.route('/')
def home():
    return render_template('page.html')

def upload_file():
    print('upload')
    file = request.files['file']
    result = predict(file)
    return result

def predict():
    # 获取上传的文件
    file = request.files['file']

    # 将文件对象转换为DataFrame对象
    df = pd.read_csv(file)

    # 选择需要预测的特征列
    selected_columns = ['max月统筹金占总比例', '月统筹金额_MAX', '月就诊次数_MAX', '本次审批金额_SUM', '月药品金额_AVG',
                        '一天去两家的天数占总天数的比']
    df_selected = df[selected_columns]

    # 进行预测操作，得到预测结果
    y_pred = model.predict(df_selected)

    return y_pred
if __name__ == "__main__":
    app.run(port=80,debug = True)