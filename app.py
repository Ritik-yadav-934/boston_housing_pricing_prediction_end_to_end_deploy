import pickle 
from flask import Flask,request, app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

# create app
app = Flask(__name__)

# Load the model
reg_model = pickle.load(open("regreModel.pkl","rb"))
scalar = pickle.load(open("scaling.pkl","rb"))
@app.route("/")
def home():
    return render_template("home.html")


def predict_api():
    data = request.json["data"]
    print(data)
    new_data = np.array(list(data.values())).reshape(1,-1)
    print("new data",new_data)
    scaled_data = scalar.transform(new_data)
    output = reg_model.predict(scaled_data)
    print("output is ",output[0])
    return jsonify(output[0])

@app.route("/predict_api",methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    scaled_data = scalar.transform(final_input)
    output = reg_model.predict(scaled_data)[0]
    return render_template("home.html",prediction_text="The predicted house price is ${:,.2f}".format(output))
    




if __name__=="__main__":
    app.run(debug=True)

