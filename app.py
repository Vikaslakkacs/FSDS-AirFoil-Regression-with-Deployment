import pickle
import flask
from flask import Flask, jsonify, request, app, url_for, render_template
from flask import Response
from flask_cors import CORS
import numpy as np



## Start the app
app= Flask(__name__)
### Loading home page

## Load the model using pickle
model= pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    ## return to home page
    return render_template('home.html')

### Capturin the data coming from appi
@app.route('/predict_airfoil_api', methods=['POST'])

def air_foil_predict_api():

    ### Save all the inputs (which is dict)into variable
    data= request.json['data']
    print(data)
    ### Picking the values from dictionary
    ### Get the dict vlaues
    ### Convert to numpy array and change the dimension
    new_data=np.expand_dims(np.array(list(data.values())), axis=0)
    output= model.predict(new_data)[0]

    return jsonify(output)

### Capturin the data coming from Webpage
@app.route('/predict', methods=['POST'])

def predict():

    ### Save all the inputs (which is dict)into variable
    data= [float(x) for x in request.form.values()]
    final_features= [np.array(data)]
    print(data)
    ### Picking the values from dictionary
    ### Get the dict vlaues
    ### Convert to numpy array and change the dimension
    #new_data=np.expand_dims(np.array(list(data.values())), axis=0)
    output= model.predict(final_features)[0]
    print(output)

    return render_template('home.html', prediction_text="Air foil pressure is {}".format( output))


if __name__=='__main__':
    app.run(debug=True)