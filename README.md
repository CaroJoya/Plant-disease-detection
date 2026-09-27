# 🌿 PlantGuard AI

### Intelligent Plant Disease Detection Using Deep Learning

PlantGuard AI is a deep-learning based image classification application that analyzes plant leaf images and predicts the most likely plant disease class.

The project combines a **Convolutional Neural Network (CNN)** with a simple **Streamlit web interface**, allowing users to upload a leaf image and receive a prediction with the model's confidence score.

---

## ✨ Features

- 🌱 **Plant disease classification** from leaf images
- 🧠 **CNN-based deep learning model**
- 🔬 Classification across **38 plant/disease categories**
- 📤 Simple image upload interface
- ⚡ Fast local prediction
- 📊 Displays prediction confidence
- 🖼️ Supports JPG, JPEG, and PNG images
- 🌿 Clean and responsive Streamlit interface
- 💾 Pre-trained `.h5` model included
- 📓 Training and experimentation notebook included
- 🧩 Modular source-code structure

---

## 🖥️ Application Preview

The application provides a clean workflow:

```text
        🌿 PlantGuard AI
  Intelligent Plant Disease Detection

              ↓

       📤 Upload Plant Leaf

              ↓

        🔍 Analyze Leaf

              ↓

       ┌───────────────────┐
       │  Analysis Result  │
       │                   │
       │  Plant: Apple     │
       │  Disease: Black   │
       │  Rot              │
       │                   │
       │  Confidence: 99%  │
       └───────────────────┘
```

The application also provides model information, supported classes, and basic guidance for obtaining better predictions.

---

## 🧠 How It Works

The application follows a straightforward image-classification pipeline:

```text
Leaf Image
    │
    ▼
Image Upload
    │
    ▼
RGB Conversion
    │
    ▼
Resize to 512 × 512
    │
    ▼
CNN Model
    │
    ▼
Class Probabilities
    │
    ▼
Highest-Probability Class
    │
    ▼
Disease / Healthy Result
```

### Prediction Process

1. The user uploads a plant leaf image.
2. The image is converted to RGB format.
3. The image is resized to **512 × 512 pixels**, matching the trained model's input shape.
4. The processed image is passed to the CNN.
5. The model generates probabilities for all supported classes.
6. The class with the highest probability is selected.
7. The application displays the predicted plant/disease category and confidence score.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **TensorFlow** | Deep learning framework |
| **Keras** | Neural network/model interface |
| **NumPy** | Numerical and array operations |
| **Pillow (PIL)** | Image processing |
| **Streamlit** | Web application interface |
| **Jupyter Notebook** | Model experimentation and analysis |
| **Git & GitHub** | Version control and project hosting |

---

## 📁 Project Structure

```text
plant-disease-detection/
│
├── app/
│   └── app.py
│
├── notebook/
│   └── plant_disease_detection.ipynb
│
├── src/
│   ├── model.py
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── saved_model/
│   └── plant_disease_model.h5
│
├── requirements.txt
├── README.md
└── .gitignore
```

### Directory Overview

#### `app/`
Contains the Streamlit application responsible for the user interface and model inference.

#### `notebook/`
Contains the Jupyter notebook used for experimentation, data analysis, model development, and evaluation.

#### `src/`
Contains the project's reusable Python modules for model definition, preprocessing, training, and prediction.

#### `saved_model/`
Contains the trained Keras model used by the Streamlit application.

#### `requirements.txt`
Contains the Python dependencies required to run the project.

---

## 🌱 Supported Classes

The model supports **38 classes** covering multiple crops and their corresponding healthy/disease categories.

### 🍎 Apple

- Apple — Apple Scab
- Apple — Black Rot
- Apple — Cedar Apple Rust
- Apple — Healthy

### 🫐 Blueberry

- Blueberry — Healthy

### 🍒 Cherry

- Cherry — Powdery Mildew
- Cherry — Healthy

### 🌽 Corn

- Corn — Cercospora Leaf Spot / Gray Leaf Spot
- Corn — Common Rust
- Corn — Northern Leaf Blight
- Corn — Healthy

### 🍇 Grape

- Grape — Black Rot
- Grape — Esca / Black Measles
- Grape — Leaf Blight
- Grape — Healthy

### 🍊 Orange

- Orange — Huanglongbing / Citrus Greening

### 🍑 Peach

- Peach — Bacterial Spot
- Peach — Healthy

### 🫑 Pepper

- Pepper — Bacterial Spot
- Pepper — Healthy

### 🥔 Potato

- Potato — Early Blight
- Potato — Late Blight
- Potato — Healthy

### 🫐 Raspberry

- Raspberry — Healthy

### 🌱 Soybean

- Soybean — Healthy

### 🎃 Squash

- Squash — Powdery Mildew

### 🍓 Strawberry

- Strawberry — Leaf Scorch
- Strawberry — Healthy

### 🍅 Tomato

- Tomato — Bacterial Spot
- Tomato — Early Blight
- Tomato — Late Blight
- Tomato — Leaf Mold
- Tomato — Septoria Leaf Spot
- Tomato — Spider Mites
- Tomato — Target Spot
- Tomato — Yellow Leaf Curl Virus
- Tomato — Mosaic Virus
- Tomato — Healthy

---

## ⚙️ Model Details

The application uses a **Convolutional Neural Network (CNN)** for image classification.

### Input

```text
Image Type : RGB
Input Size : 512 × 512 × 3
```

### Output

```text
Number of Classes : 38
Output            : Class probabilities
```

The predicted class is selected using the highest probability produced by the model.

The model is stored as:

```text
saved_model/plant_disease_model.h5
```

### Custom Layer Compatibility

The saved model contains a custom `FixedDropout` layer implementation. The Streamlit application defines the corresponding compatibility class and supplies it through Keras `custom_objects` while loading the model.

This allows the saved model to be loaded correctly during inference.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/agg-ayush/plant-disease-detection.git
```

Move into the project directory:

```bash
cd plant-disease-detection
```

---

### 2. Create a virtual environment

It is recommended to use a separate virtual environment for the project.

#### Windows

```bash
python -m venv plant-env
```

Activate it:

```bash
plant-env\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv plant-env
```

Activate it:

```bash
source plant-env/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If TensorFlow is being installed on Windows, make sure the Python version is compatible with the TensorFlow version specified by the project dependencies.

---

## ▶️ Running the Application

The Streamlit application is located inside the `app` directory.

Move into the application directory:

```bash
cd app
```

Then run:

```bash
python -m streamlit run app.py
```

On Windows, if you are using a specific virtual environment interpreter:

```bash
F:\plant-env\Scripts\python.exe -m streamlit run app.py
```

Streamlit will provide a local URL, normally similar to:

```text
http://localhost:8501
```

Open the URL in your browser.

---

## 📸 Using the Application

### Step 1 — Upload an image

Click the upload area and select a plant leaf image.

Supported formats:

```text
.jpg
.jpeg
.png
```

### Step 2 — Preview

The uploaded image will be displayed in the application.

### Step 3 — Analyze

Click:

```text
🔍 Analyze Leaf
```

### Step 4 — View the result

The application displays:

- Plant category
- Predicted disease/healthy class
- Model confidence
- Prediction details

Example:

```text
Analysis Result

Apple
Black Rot

Model Confidence
████████████████████ 100.0%
```

---

## 📊 Understanding Confidence

The displayed confidence is the probability assigned by the model to its selected class.

For example:

```text
Black Rot
Confidence: 94.7%
```

means that the model assigned approximately **94.7% probability to the Black Rot class for that particular input**.

A high model confidence should not automatically be interpreted as guaranteed real-world correctness. Image quality, lighting, background, leaf condition, and similarity to the training data can affect predictions.

---

## 🧪 Model Development

The project includes a Jupyter notebook containing the model-development workflow.

Notebook:

```text
notebook/plant_disease_detection.ipynb
```

The notebook can be used to explore:

- Image dataset processing
- Data preparation
- CNN model development
- Model training
- Validation
- Performance evaluation
- Prediction experiments

---

## 🔧 Source Modules

The `src` directory separates important parts of the machine-learning workflow.

### `model.py`

Contains model-related definitions.

### `preprocess.py`

Contains image/data preprocessing functionality.

### `train.py`

Contains training-related functionality.

### `predict.py`

Contains prediction-related functionality.

This separation makes the project easier to understand, maintain, and extend.

---

## 🎨 Streamlit Application

The main application is implemented in:

```text
app/app.py
```

The interface includes:

- PlantGuard AI branding
- Image upload component
- Image preview
- Analyze button
- Prediction result card
- Confidence progress indicator
- Model information
- Supported-class information
- User guidance for better images

The interface is designed to keep the machine-learning workflow simple enough for a user without requiring them to interact directly with Python or the model.

---

## 📌 Recommended Input Images

For better predictions, use images that:

- Clearly show the plant leaf
- Have sufficient lighting
- Have reasonable image quality
- Keep the leaf visible and relatively large in the frame
- Avoid excessive blur
- Avoid heavily obstructed leaves

The model performs image classification and does not replace professional agricultural diagnosis.

---

## ⚠️ Limitations

This project is intended for **educational and experimental purposes**.

Some limitations include:

- Predictions depend on the quality of the input image.
- The model only recognizes the classes included in its training setup.
- Unseen diseases may be incorrectly classified as one of the supported classes.
- Similar-looking symptoms can result in incorrect predictions.
- Model confidence does not guarantee real-world diagnostic accuracy.
- Environmental conditions and camera quality can affect image appearance.

For real agricultural decisions, predictions should be verified using appropriate expert or laboratory assessment.

---

## 🔮 Future Improvements

Possible extensions include:

- 📱 Mobile-friendly deployment
- 📷 Real-time camera-based leaf detection
- 🌍 Multi-language support
- 📈 Prediction history and analytics
- 🗃️ Database integration
- ☁️ Cloud deployment
- 🧠 Model architecture improvements
- 🔍 Explainable AI using techniques such as Grad-CAM
- 📊 Detailed disease information pages
- 🌱 Crop-specific recommendations
- 🔄 Continuous model improvement with additional datasets

---

## 💡 Project Workflow

```text
                ┌─────────────────────┐
                │    Leaf Image       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Preprocessing     │
                │   RGB + Resize      │
                │    512 × 512        │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │     CNN Model       │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Class Probabilities │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Highest Probability │
                │       Class         │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Streamlit Result    │
                │ + Confidence Score  │
                └─────────────────────┘
```

---

## 🔐 Privacy

Images uploaded to the application are processed locally when running the project on your own machine.

The application does not require an external image-analysis API or cloud AI service for its core prediction workflow.

---

## 📚 Learning Outcomes

This project demonstrates practical concepts in:

- Machine Learning
- Deep Learning
- Convolutional Neural Networks
- Image Classification
- Image Preprocessing
- TensorFlow and Keras
- Model Inference
- Python Development
- Streamlit Application Development
- Git and GitHub
- Modular Project Structure

---

## 🏁 Quick Start

For users who already have Python and the required dependencies installed:

```bash
git clone https://github.com/agg-ayush/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
cd app
python -m streamlit run app.py
```

Then open the Streamlit URL in your browser, upload a plant leaf image, and click **Analyze Leaf**.

---

## 📄 License

This project is provided for educational and research purposes.

If you reuse or extend the project, review the licenses and usage terms of the datasets, libraries, and model components involved.

---

## 🌿 PlantGuard AI

> **Upload a leaf. Analyze the image. Understand the prediction.**

Built with **Python, TensorFlow, Keras, NumPy, Pillow, and Streamlit**.
