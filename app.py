from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

with open("svm_ckd_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("svm_ckd_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

feature_columns = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
    'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
    'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'pedal_edema', 'aanemia'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            print(request.form)  # Debug
            input_data = {
                'age': float(request.form['age']),
                'blood_pressure': float(request.form['blood_pressure']),
                'specific_gravity': float(request.form['specific_gravity']),
                'albumin': float(request.form['albumin']),
                'sugar': float(request.form['sugar']),
                'red_blood_cells': request.form['red_blood_cells'],
                'pus_cell': request.form['pus_cell'],
                'pus_cell_clumps': request.form['pus_cell_clumps'],
                'bacteria': request.form['bacteria'],
                'blood_glucose_random': float(request.form['blood_glucose_random']),
                'blood_urea': float(request.form['blood_urea']),
                'serum_creatinine': float(request.form['serum_creatinine']),
                'sodium': float(request.form['sodium']),
                'potassium': float(request.form['potassium']),
                'hemoglobin': float(request.form['hemoglobin']),
                'packed_cell_volume': float(request.form['packed_cell_volume']),
                'white_blood_cell_count': float(request.form['white_blood_cell_count']),
                'red_blood_cell_count': float(request.form['red_blood_cell_count']),
                'hypertension': request.form['hypertension'],
                'diabetes_mellitus': request.form['diabetes_mellitus'],
                'coronary_artery_disease': request.form['coronary_artery_disease'],
                'appetite': request.form['appetite'],
                'pedal_edema': request.form['pedal_edema'],
                'aanemia': request.form['aanemia']
            }
            df_input = pd.DataFrame([input_data])
            df_input = df_input[feature_columns]

            mappings = {
                'red_blood_cells': {'normal': 0, 'abnormal': 1},
                'pus_cell': {'normal': 0, 'abnormal': 1},
                'pus_cell_clumps': {'notpresent': 0, 'present': 1},
                'bacteria': {'notpresent': 0, 'present': 1},
                'hypertension': {'no': 0, 'yes': 1},
                'diabetes_mellitus': {'no': 0, 'yes': 1},
                'coronary_artery_disease': {'no': 0, 'yes': 1},
                'appetite': {'poor': 0, 'good': 1},
                'pedal_edema': {'no': 0, 'yes': 1},
                'aanemia': {'no': 0, 'yes': 1}
            }
            for col, mapping in mappings.items():
                df_input[col] = df_input[col].map(mapping)

            scaled = scaler.transform(df_input)
            prediction = model.predict(scaled)[0]
            probability = model.predict_proba(scaled)[0][1]


            print("==== INPUT FORM DATA ====")
            print(request.form)
            print("Processed input_data:", input_data)
            print("Preprocessed dataframe:\n", df_input)
            print("Prediction:", prediction)
            print("Probability:", probability)

        except Exception as e:
            error = str(e)
            print("เกิดข้อผิดพลาด: ", error)

    return render_template("index_ckd-1co.html",
                           prediction=prediction,
                           probability=probability,
                           error=error)

    # return render_template("result.html",
    #                        prediction=prediction,
    #                        probability=probability,
    #                        error=error)

if __name__ == "__main__":
    app.run(debug=True)
