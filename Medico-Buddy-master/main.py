

from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
app.static_folder = 'static'

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

filename = 'heart.pkl'
model = pickle.load(open(filename, 'rb'))

filename = 'insurance.pkl'
regressor = pickle.load(open(filename, 'rb'))

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/covid', methods=["GET", "POST"])
def covid():
    if request.method == "POST":
        myDict = request.form
        Fever = int(myDict['Fever'])
        Age = int(myDict['Age'])
        BodyPain = int(myDict['BodyPain'])
        RunnyNose = int(myDict['RunnyNose'])
        DiffBreath = int(myDict['DiffBreath'])
        ChestPain = int(myDict['ChestPain'])

        inputFeatures = [Fever, BodyPain, Age,
                         RunnyNose, DiffBreath, ChestPain]
        InfProba = clf.predict_proba([inputFeatures])[0][1]
        print(InfProba)
        return render_template('show.html', inf=round(InfProba*100))
    return render_template('covid.html')
    # return 'Hello, World!' + str(InfProba)


@app.route('/diabetes', methods=["POST", "GET"])
def diabetes():
    if request.method == 'POST':
        myDict2 = request.form
        preg = int(myDict2['pregnancies'])
        glucose = int(myDict2['glucose'])
        bp = int(myDict2['bloodpressure'])
        st = int(myDict2['skinthickness'])
        insulin = int(myDict2['insulin'])
        bmi = float(myDict2['bmi'])
        dpf = float(myDict2['dpf'])
        age = int(myDict2['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('diashow.html', prediction=my_prediction)
    return render_template('diabetes.html')


@app.route('/heart', methods=["POST", "GET"])
def heart():
    if request.method == 'POST':
        myDict3 = request.form
        age = int(myDict3['age'])
        sex = int(myDict3['sex'])
        cp = int(myDict3['cp'])
        trestbps = int(myDict3['trestbps'])
        chol = int(myDict3['chol'])
        fbs = float(myDict3['fbs'])
        restecg = float(myDict3['restecg'])
        thalach = int(myDict3['thalach'])
        exang = int(myDict3['exang'])
        oldpeak = int(myDict3['oldpeak'])
        slope = int(myDict3['slope'])
        ca = int(myDict3['ca'])
        thal = int(myDict3['thal'])

        data1 = np.array([[age, sex, cp, trestbps, chol, fbs,
                           restecg, thalach, exang, oldpeak, slope, ca, thal]])
        my_prediction1 = model.predict(data1)

        return render_template('heartshow.html', prediction1=my_prediction1)
    return render_template('heart.html')


@app.route('/insurance', methods=["POST", "GET"])
def insurance():
    if request.method == 'POST':
        myDict4 = request.form
        age = int(myDict4['age'])
        sex = int(myDict4['sex'])
        bmi = int(myDict4['bmi'])
        children = int(myDict4['children'])
        smoker = int(myDict4['smoker'])
        region = float(myDict4['region'])

        data = np.array([[age, sex, bmi, children, smoker, region]])
        my_prediction3 = regressor.predict(data)

        return render_template('insurshow.html', prediction3=my_prediction3)
    return render_template('insurance.html')


if __name__ == "__main__":
    app.run(debug=True)
