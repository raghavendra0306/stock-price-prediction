from flask import Flask, request, render_template  # Import Flask for web app and utilities
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
from sklearn.linear_model import LinearRegression  # Import Linear Regression model
from sklearn.model_selection import train_test_split  # Import function for splitting data

# Initialize the Flask app
app = Flask(__name__)

# Load the dataset and preprocess
tesla = pd.read_csv('tesla.csv')  # Load the Tesla stock dataset
tesla['Date'] = pd.to_datetime(tesla['Date'])  # Convert the Date column to datetime format

# Select input (features) and target (label)
X = tesla[['Open', 'High', 'Low']]  # Features: Open, High, Low prices
Y = tesla['Close']  # Target: Close price

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

# Initialize and train the Linear Regression model
lm = LinearRegression()
lm.fit(X_train, Y_train)

# Define the home route for the web app
@app.route('/')
def home():
    # Render the home page (index.html)
    return render_template('index.html')

# Define the predict route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract user inputs from the form on the webpage
        open_price = float(request.form['open'])  # Get the 'open' price
        high_price = float(request.form['high'])  # Get the 'high' price
        low_price = float(request.form['low'])  # Get the 'low' price

        # Create a DataFrame for the input data
        input_data = pd.DataFrame({'Open': [open_price], 'High': [high_price], 'Low': [low_price]})

        # Predict the close price using the trained model
        predicted_close = lm.predict(input_data)[0]

        # Generate a recommendation based on the predicted close price
        if predicted_close > open_price:
            recommendation = "Buy the Stock"
        else:
            recommendation = "Do not buy the Stock"

        # Render the result back to the webpage
        return render_template(
            'index.html',
            prediction_text=f'Predicted Close Price: ${predicted_close:.2f}',
            recommendation_text=recommendation
        )
    except Exception as e:
        # Handle any errors and display a message on the webpage
        return render_template('index.html', prediction_text=f'Error: {e}')

# Run the Flask web app
if __name__ == "__main__":
    app.run(debug=True)

