from flask import Flask, render_template, request
import pandas as pd
import re
import joblib  # For loading model and scaler

app = Flask(__name__)

# Load your model and scaler here
try:
    model = joblib.load('model.pkl')  # Replace with your model filename
    scaler = joblib.load('scaler.pkl')  # Replace with your scaler filename
except:
    model = None
    scaler = None
    print("⚠️ Warning: Model or Scaler not loaded.")

# Define feature names as per your HTML input 'name' attributes
FEATURE_NAMES = ["u_q", "coolant", "stator_winding", "u_d", "stator_tooth",
                 "motor_speed", "i_d", "i_q", "stator_yoke", "torque"]


@app.route('/')
def home():
    return render_template('manual_predict.html')

@app.route('/y_predict', methods=['POST'])
def y_predict():
    try:
        values = []
        for feature in FEATURE_NAMES:
            raw_val = request.form.get(feature, '').strip()
            print(f"{feature} -> raw input: '{raw_val}'")  # Debugging

            if raw_val == '':
                raise ValueError(f"Missing value for '{feature}'")

            cleaned = raw_val.replace(',', '.').strip()

            # Regex: Optional -, digits, optional decimal part
            if not re.match(r'^-?\d+(\.\d+)?$', cleaned):
                raise ValueError(f"Invalid number format for '{feature}': '{raw_val}'")

            val = float(cleaned)
            values.append(val)

        input_df = pd.DataFrame([values], columns=FEATURE_NAMES)
        print("📥 Received Input Data:")
        print(input_df)

        # Prediction
        if model is not None and scaler is not None:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            prediction_text = f"🔍 Predicted Rotor Temperature: {prediction:.2f}°C"
        else:
            # Demo logic
            prediction = values[0] + values[5] / 100 + abs(values[7]) * 0.5
            prediction_text = f"⚠️ Model not loaded. Demo prediction: {prediction:.2f}°C"

        return render_template('manual_predict.html',
                               prediction_text=prediction_text,
                               **dict(zip(FEATURE_NAMES, [str(v) for v in values])))

    except Exception as e:
        print("❌ Error during prediction:", e)
        import traceback
        traceback.print_exc()

        return render_template('manual_predict.html',
                               prediction_text="❌ Please enter valid numeric inputs.",
                               **request.form)

if __name__ == '__main__':
    app.run(debug=True)
