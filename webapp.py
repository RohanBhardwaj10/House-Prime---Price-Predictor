from flask import Flask, request, render_template
import joblib
import pandas as pd

# Create the Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('best_rf_model.pkl')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data31 = request.form.to_dict()
    data_df = pd.DataFrame([data31])

    # Print the DataFrame to debug
    print("Original DataFrame:")
    print(data_df.head())

    # Convert data types as needed
    data_df = data_df.apply(pd.to_numeric, errors='coerce')

    # Print DataFrame after type conversion
    print("DataFrame after conversion to numeric:")
    print(data_df.head())

    # Define the expected feature names in the order the model was trained with
    expected_columns = [
        'area', 'latitude', 'longitude', 'Bedrooms', 'Bathrooms', 'Balcony',
        'parking', 'Lift', 'Price_sqft', 'Furnished', 'Semi-Furnished',
        'Unfurnished', 'New Property', 'Resale', 'Flat', 'Individual House'
    ]

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in data_df.columns:
            data_df[col] = 0  # Default value for missing columns

    # Reorder columns to match the trained model's expected order
    data_df = data_df[expected_columns]

    # Print DataFrame after reordering columns
    print("DataFrame after reordering columns:")
    print(data_df.head())

    prediction = model.predict(data_df)
    print("Prediction result:")
    print(prediction)

    # Return the prediction
    return render_template('index.html', prediction_text=f'Predicted House Price: INR : {prediction[0]:,.2f}')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
