from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

nn_model = joblib.load('nn_model.pkl')

features = ['Q1', 'Q2', 'A1', 'Q3', 'Q4', 'Midterm']
targets = ['Q5', 'A2', 'Q6', 'Q7', 'Q8', 'Final', 'Total']

feature_ranges = {
    'Q1': 30,
    'Q2': 49,
    'A1': 100,
    'Q3': 30,
    'Q4': 15,
    'Midterm': 35
}

target_ranges = {
    'A2': 100,
    'Q6': 32,
    'Q7': 24,
    'Q8': 40,
    'Final': 40,
    'Q5': 45,
    'Total': 100
}

class MockScaler:
    def transform(self, X):
        return X

scaler = MockScaler()

def predict_targets(input_features):
    if len(input_features) != len(features):
        raise ValueError(f"Expected {len(features)} features, but got {len(input_features)}.")

    normalized_features = [input_features[i] / feature_ranges[features[i]] for i in range(len(features))]
    input_df = pd.DataFrame([normalized_features], columns=features)
    input_scaled = scaler.transform(input_df)
    predictions = nn_model.predict(input_scaled)

    denormalized_predictions = predictions[0].copy()
    for i, target in enumerate(targets):
        if target in target_ranges:
            denormalized_predictions[i] *= target_ranges[target]
            denormalized_predictions[i] = min(max(0, denormalized_predictions[i]), target_ranges[target])

    return dict(zip(targets, denormalized_predictions))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            input_features = [
                float(request.form.get(feature)) for feature in features
            ]

            prediction = predict_targets(input_features)
        except Exception as e:
            error = str(e)

    return render_template('simple_index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0' , port='50001' , debug=True)
