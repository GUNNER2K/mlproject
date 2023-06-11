
from flask import Flask , request ,render_template , jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData , PredictPipeline

app = Flask(__name__)

@app.route('/predict' , methods = ['POST'])
def predict():
    json_ = request.json 
    dataframe = pd.DataFrame(json_)
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(dataframe)
    return str(results).split(' ')
if __name__ == '__main__':
    app.run(debug=True , port=5000)