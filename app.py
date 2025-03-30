from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib 
from sklearn.preprocessing import LabelEncoder,StandardScaler
import sklearn

app = Flask(__name__)
CORS(app)

@app.route('/',methods=["GET"])
def hello():
    return "Hello world"
@app.route('/add', methods=['POST'])
def add1():
    # print("Received Body:", request.data)
    print("Received Headers:", request.headers.get('Content-Type'))
    # if request.headers.get('Content-Type') != 'application/json; charset=utf-8':
    #     return jsonify({'error': 'Content-Type must be application/json'}), 400

    try:
        data = request.get_json()
        print(data)
        # if data is None:
        #     return jsonify({'error': 'Invalid JSON data'}), 400

        values=data['values']
        sc=StandardScaler()
        values=sc.fit_transform([values])
        if(data['disease']==0):
            model_test = joblib.load('ExtraTrees_LB.joblib')
            # print(sklearn.__version__)
            prediction=model_test.predict(values)
        elif(data['disease']==1):
            model_test = joblib.load('ExtraTrees_NB.joblib')
            # sc=StandardScaler()
            prediction=model_test.predict(values)
        elif(data['disease']==2):
            model_test = joblib.load('LinearReg_GD.joblib')
            # sc=StandardScaler()
            prediction=model_test.predict(values)
        elif(data['disease']==3):
            model_test = joblib.load('ExtraTrees_SR.joblib')
            # sc=StandardScaler()
            prediction=model_test.predict(values)
        elif(data['disease']==4):
            model_test = joblib.load('BayesRi_SB.joblib')
            sc=StandardScaler()
            prediction=model_test.predict(values)
        elif(data['disease']==5):
            model_test = joblib.load('LinearReg_BS.joblib')
            prediction=model_test.predict(values)
        data['prediction']=prediction[0]
        return jsonify({'message': 'Success', 'received_data': data}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
