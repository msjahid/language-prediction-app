import warnings

warnings.filterwarnings("ignore")  # Suppress all warnings

from flask import Flask, request, render_template
import pickle
import re
import numpy as np

# Initialize the Flask application
app = Flask(__name__, static_folder='core/static', template_folder='core/templates')


# Define custom tokenizer explicitly here
def custom_tokenizer(text):
    return text.split()  # Simple space-based tokenizer


# Load the serialized components
try:
    with open('core/models/language_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('core/models/vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    with open('core/models/label_encoder.pkl', 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)

except Exception as e:
    print(f"Error loading model components: {e}")
    model = vectorizer = encoder = None


# Define the preprocessing function
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    text = re.sub(r'\d+', '', text)
    return text


@app.route('/', methods=['GET', 'POST'])
def predict_language():
    prediction = ''
    confidence = ''
    if request.method == 'POST':
        input_text = request.form['text']
        if input_text.strip() == '':
            prediction = "Please enter valid text."
        else:
            try:
                # Processing the text and making predictions
                processed_text = preprocess_text(input_text)
                text_vector = vectorizer.transform([processed_text])
                predicted_proba = model.predict_proba(text_vector)
                max_proba = np.max(predicted_proba)
                confidence_threshold = 0.7  # Set your desired threshold

                if max_proba < confidence_threshold:
                    prediction = ("We are unable to confidently detect the language. We will work on improving our "
                                  "model.")
                else:
                    predicted_label = model.predict(text_vector)
                    language = encoder.inverse_transform(predicted_label)[0]
                    prediction = f"The predicted language is: {language}"
                    confidence = f"Confidence Level: {max_proba * 100:.0f}%"
            except Exception as e:
                # If any error occurs, render the error page
                return render_template('error.html', error_message=f"An error occurred: {str(e)}")

    return render_template('index.html', prediction=prediction, confidence=confidence)


@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error_message="Internal Server Error")


@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error_message="Page Not Found")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
