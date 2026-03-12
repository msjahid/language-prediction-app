# LangDetect - Language Detection System

LangDetect is a Flask-based web application designed to detect the language of input text using machine learning. The app uses a trained Multinomial Naive Bayes model to provide instant language predictions with **98% accuracy** across **17 supported languages**. It is deployed on Railway for ease of access.

Check out the live app here: [Language Prediction App](https://language-prediction-app-production.up.railway.app/)

---

## Features

- User-friendly text input interface for language detection.
- Detects language using a Multinomial Naive Bayes model with 98% accuracy.
- Supports 17 different languages with flag-based visual display.
- Displays a custom message when the input language is not in the supported list.
- Shows confidence level indicating how certain the model is about its prediction.
- Handles errors gracefully with descriptive error messages.
- Deployed on Railway for seamless access.

---

## Supported Languages

| Flag | Language   | Flag | Language  | Flag | Language |
| ---- | ---------- | ---- | --------- | ---- | -------- |
| 🇺🇸   | English    | 🇫🇷   | French    | 🇪🇸   | Spanish  |
| 🇵🇹   | Portuguese | 🇮🇹   | Italian   | 🇷🇺   | Russian  |
| 🇸🇪   | Swedish    | 🇮🇳   | Malayalam | 🇳🇱   | Dutch    |
| 🇸🇦   | Arabic     | 🇹🇷   | Turkish   | 🇩🇪   | German   |
| 🇮🇳   | Tamil      | 🇩🇰   | Danish    | 🇮🇳   | Kannada  |
| 🇬🇷   | Greek      | 🇮🇳   | Hindi     |      |          |

---

## Technologies Used

- **Python**: Core programming language.
- **Flask**: Web framework for building the app.
- **Machine Learning**: Multinomial Naive Bayes model for language detection.
- **Scikit-learn**: Model training, evaluation, and serialization.
- **HTML/CSS/JavaScript**: Frontend templates and styling.
- **Railway**: Cloud platform for app deployment.

---

## Folder Structure

```plaintext
.
├── Dockerfile                  # Docker instructions for building the app
├── app.py                      # Flask application entry point
├── core
│   ├── __init__.py             # Core package initializer
│   ├── model_storage
│   │   └── mnb_model.pkl       # Pre-trained Multinomial Naive Bayes model
│   ├── models.py               # Model loading and prediction logic
│   ├── request.py              # Input parsing and validation
│   ├── routes.py               # Flask app routes
│   ├── static
│   │   ├── css
│   │   │   └── style.css       # Styling for the app
│   │   └── images
│   │       ├── favicon.ico     # App icon
│   │       └── logo.png        # Logo for branding
│   └── templates
│       ├── error.html          # Error page template
│       ├── index.html          # Language detection input form
│       └── prediction.html     # Prediction result display
├── project_images              # Documentation screenshots
│   ├── home.png                # Screenshot of the Home Page
│   ├── prediction.png          # Screenshot of the Prediction Page
│   └── error.png               # Screenshot of the Error Page
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker Compose configuration
└── README.md                   # Project documentation
```

---

## INSTALLATION GUIDE LOCAL

### Step 1: Clone the repository

```bash
git clone https://github.com/msjahid/langdetect-app.git
cd langdetect-app
```

### Step 2: Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Flask app

```bash
python app.py
```

### Step 5: Open the app in your browser

```bash
http://127.0.0.1:5000/
```

---

## INSTALLATION GUIDE DOCKER

### Step 1: Prerequisites

- **Docker and Docker Compose installed on your machine.**
- **Your project structured properly (as defined earlier).**

### Step 2: Build the Docker Image

From the project directory, run the following command to build the Docker image:

```bash
docker-compose build
```

### Step 3: Run the Application

Start the application using:

```bash
docker-compose up
```

This will:

- **Build the Docker container if it hasn't been built already.**
- **Start the Flask application inside the container.**

### Step 4: Access the Application

Once the container is running, open your browser and navigate to: Check your port number

```bash
http://127.0.0.1:4000
```

---

## DEPLOYMENT ON RAILWAY

### Step 1: Sign up or log in to Railway

Visit [https://railway.app/](https://railway.app/) and create an account.

### Step 2: Create a new project

- Click "New Project" → "Deploy from GitHub repo".
- Select your repository (e.g., `langdetect-app`).

### Step 3: Configure the deployment

- Railway auto-detects the Procfile and runtime.txt.
- Ensure dependencies in requirements.txt are accurate.

### Step 4: Deploy the app

- Wait for the build and deployment to complete.

### Step 5: Access the app

Railway provides a unique URL (e.g., `https://language-prediction-app-production.up.railway.app/`).

---

## INPUT VALIDATION RULES

1. **Text Input**: Cannot be empty or whitespace only.
2. **Text Length**: Must be at least 3 characters for reliable detection.
3. **Language Support**: If the detected language is outside the 17 supported languages, a custom "Unable to detect language" message is displayed.
4. **Confidence Level**: The app reports the model's confidence percentage alongside the prediction.

---

## PAGE PREVIEW SCREENSHOTS

### 1. Home Page

- Displays the language detection input form.
- Includes a text area to enter text for detection.
- Shows the full list of 17 supported languages with flag icons.

![Home Page](https://raw.githubusercontent.com/msjahid/language-prediction-app/refs/heads/main/project-image/home.png)

### 2. Prediction Page

- Displays the detected language result.
- Shows the confidence level of the prediction.
- Displays a custom message if the language is unsupported.

![Prediction Page](https://raw.githubusercontent.com/msjahid/language-prediction-app/refs/heads/main/project-image/prediction.png)

### 2. Prediction Page Non listed language

- Displays the non listed language result.

![Non listed Page](https://raw.githubusercontent.com/msjahid/language-prediction-app/refs/heads/main/project-image/non_listed_language.png)

### 3. Error Page

- Displays descriptive error messages for invalid or empty inputs.
- Includes a visual illustration for better UX.

![Error Page](https://raw.githubusercontent.com/msjahid/language-prediction-app/refs/heads/main/project-image/error.png)

---

## EXAMPLE WORKFLOW

### INPUT

```
Text: "Bonjour, comment ça va?"
```

### OUTPUT (Detected)

```
Detected Language: French 🇫🇷
Confidence: 100%
```

### OUTPUT (Unsupported Language)

```
ইরানে হামলা বন্ধ করে আলোচনার টেবিলে ফিরে আসতে যুক্তরাষ্ট্র ও ইসরায়েলের প্রতি আহ্বান জানিয়েছে রাশিয়া।
```

```
We are unable to confidently detect the language. We will work on improving our model.
```

---

## CONTRIBUTING

### Steps to contribute:

1. Fork the repository.

2. Create a new branch for your feature:

   ```bash
   git checkout -b feature-name
   ```

3. Commit your changes:

   ```bash
   git commit -m "Add new feature"
   ```

4. Push to your branch:

   ```bash
   git push origin feature-name
   ```

5. Submit a pull request for review.

---

## LICENSE

This project is open-source and available under the MIT License.

---

## CONTACT

- **Author**: Jahid Hasan
- **GitHub**: [https://github.com/msjahid](https://github.com/msjahid)
- **Email**: [msjahid.ai@gmail.com](mailto:msjahid.ai@gmail.com)
