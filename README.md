# IMDB Movie Review Sentiment Analysis

[![PyTorch](https://img.shields.io/badge/Made%20with-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Field-Deep%20Learning-FF69B4)](https://en.wikipedia.org/wiki/Deep_learning)
[![Natural Language Processing](https://img.shields.io/badge/Task-NLP-blue)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Streamlit](https://img.shields.io/badge/Deployment-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![GitHub last commit](https://img.shields.io/github/last-commit/mobadara/imdb-movie-review-sentiment-analysis)](https://github.com/mobadara/imdb-movie-review-sentiment-analysis/commits/main)
[![Repo Size](https://img.shields.io/github/repo-size/mobadara/imdb-movie-review-sentiment-analysis)](https://github.com/mobadara/imdb-movie-review-sentiment-analysis)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a deep learning project focused on **sentiment analysis** of movie reviews from the IMDB dataset. The goal is to classify movie reviews as either **positive** or **negative** using a **PyTorch**-based neural network. This project demonstrates skills in natural language processing (NLP), deep learning model development, and practical application of PyTorch.

---

## Project Overview

Sentiment analysis is a subfield of NLP that deals with identifying and extracting subjective information from text. In this project, an end-to-end pipeline was developed to:

1.  **Process raw text data:** Convert unstructured movie review text into a numerical format suitable for deep learning models.
2.  **Build a recurrent neural network (RNN) model:** Design and implement a sentiment classification model using PyTorch's capabilities, specifically leveraging **Long Short-Term Memory (LSTM)** networks, which are well-suited for sequence data like text.
3.  **Train and evaluate the model:** Train the neural network on a large dataset of movie reviews and evaluate its performance in classifying sentiment accurately.
4.  **Deploy the model:** Create a user-friendly web interface using **Streamlit** to allow interactive sentiment prediction.

---

## Dataset

The project utilizes the **IMDB Movie Review Dataset**, which consists of 50,000 highly polarized movie reviews. This dataset is well-balanced, with an equal number of positive and negative reviews, making it ideal for binary sentiment classification tasks.

---

## Key Technologies Used

* **PyTorch:** The primary deep learning framework used for building, training, and evaluating the neural network.
* **Python:** The programming language for all development.
* **pandas:** For data loading and initial data manipulation.
* **NLTK (Natural Language Toolkit):** Used for various text preprocessing steps such as tokenization, stop word removal, and lemmatization.
* **Streamlit:** For deploying the trained model as an interactive web application.
* **scikit-learn:** For data splitting and performance metrics.

---

## Project Structure

The repository typically includes:

* **`notebooks/`**: Jupyter notebooks detailing the data preprocessing, model architecture, training process, and evaluation. This is where the core deep learning development resides.
* **`src/`**: Python scripts for reusable components like data utilities, model definitions, or custom preprocessing functions.
* **`models/`**: Directory to store trained model weights and possibly other artifacts like vocabulary files.
* **`app.py`**: The main Streamlit application file for model deployment.
* **`data/`**: (Optional) Directory to store the raw or processed dataset.
* **`requirements.txt`**: Lists all necessary Python dependencies for reproducibility.

---

## Getting Started

To set up and run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mobadara/imdb-movie-review-sentiment-analysis.git](https://github.com/mobadara/imdb-movie-review-sentiment-analysis.git)
    cd imdb-movie-review-sentiment-analysis
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Explore the notebooks:** Navigate to the `notebooks/` directory to see the detailed steps of the project from data loading to model training and evaluation.
5.  **Run the Streamlit application:**
    After training your model and saving the necessary artifacts, you can run the deployment app:
    ```bash
    streamlit run app.py
    ```

---

## Model Architecture Highlights

The core of this project is an **LSTM-based neural network**. The architecture typically involves:

* An **Embedding Layer**: Converts input words into dense vector representations.
* **LSTM Layers**: Capture sequential dependencies and contextual information within the movie reviews. Bidirectional LSTMs are often employed for richer context.
* **Dropout Layers**: Used to prevent overfitting during training.
* **Fully Connected Layers**: Map the LSTM's output to the final sentiment prediction.

---

## Results

The model's performance was evaluated using standard classification metrics such as **accuracy**, **precision**, **recall**, and **F1-score** on a held-out test set. The results demonstrate the effectiveness of the chosen deep learning approach for this binary sentiment classification task.

---

## Live Demo (Deployment)

A live demo of the sentiment analysis model is available via Streamlit. You can interact with the deployed model by entering your own movie reviews and getting instant sentiment predictions.

*(You might add a link to your Streamlit Cloud or Hugging Face Spaces deployment here if you host it, e.g., `[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app-url.streamlit.app/)`)*

---

## Author

**mobadara**

---
