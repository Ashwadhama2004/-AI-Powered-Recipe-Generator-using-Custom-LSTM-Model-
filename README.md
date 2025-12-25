
# 🍽️ Image-to-Recipe Generation – AI Cooking Assistant

Welcome to the **Image-to-Recipe Generation Project**!
This project focuses on generating **complete cooking recipes directly from food images** using **deep learning**. It combines **computer vision** and **natural language generation** through a **CNN + Seq2Seq (LSTM) architecture**, turning pixels into ingredients and cooking steps.

👉 **Core Idea**:
Upload a food image → AI understands the dish → Generates ingredients + step-by-step recipe.

---

## 🔍 Project Overview

This system:

* Takes a **food image** as input.
* Extracts **visual features** using a pre-trained CNN.
* Generates a **human-readable recipe** using a Seq2Seq model.
* Outputs ingredients and cooking instructions in natural language.

The project is inspired by real-world AI cooking assistants and research benchmarks like **Recipe1M**.

---

## 🧠 Methodology (High-Level)

1. **Dataset Preparation**

   * Food images paired with recipes (ingredients + instructions).
   * Cleaned, structured, and loaded into the pipeline.

2. **Image Preprocessing**

   * Resize (e.g., 224×224), normalize, and augment images.

3. **Feature Extraction**

   * Pre-trained CNN (ResNet-50 / VGG-16) extracts visual embeddings.

4. **Text Processing**

   * Tokenization, vocabulary building, embeddings for recipe text.

5. **Seq2Seq Architecture**

   * Encoder: Image feature vector.
   * Decoder: LSTM/GRU generating recipe word-by-word.

6. **Training**

   * Cross-entropy loss with teacher forcing.
   * Optimized using Adam.

7. **Inference**

   * New image → Generated ingredients + cooking steps.

8. **Evaluation**

   * BLEU / ROUGE metrics for recipe quality and coherence.

---

## 🛠️ Tech Stack

* **Python** 🐍
* **Deep Learning**:

  * TensorFlow / PyTorch
  * CNN (ResNet / VGG)
  * LSTM-based Seq2Seq
* **NLP**:

  * Tokenization
  * Word embeddings
* **Computer Vision**:

  * OpenCV
  * Image augmentation
* **Evaluation**:

  * BLEU, ROUGE

---

## ⚙️ Features

* 🖼️ **Food Image Understanding**

  * Learns visual patterns of dishes.

* 📜 **Automatic Recipe Generation**

  * Generates ingredients and step-by-step instructions.

* 🔁 **End-to-End Pipeline**

  * Image → Encoder → Decoder → Recipe text.

* 📊 **Model Evaluation**

  * Quantitative metrics for text similarity and fluency.

---

## 🚀 How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/image-to-recipe-generation.git
cd image-to-recipe-generation
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train.py
```

### 5. Generate Recipe from Image

```bash
python infer.py --image sample_food.jpg
```

---

## 📊 Results & Observations

* The model learns strong visual–text associations.
* Generates **coherent ingredient lists** and **logical cooking steps**.
* Performance improves significantly with:

  * Larger datasets
  * Attention mechanisms
  * Better text embeddings

---

## 🔮 Future Improvements

* Add **attention mechanism** for better alignment.
* Use **Transformer-based decoders**.
* Support **multi-image inputs**.
* Deploy via **Streamlit / Gradio** as a web app.
* Improve diversity and creativity of recipes.

---

## 📚 References

* Recipe1M Dataset
* CNN + Seq2Seq literature
* Image Captioning and Vision-Language Models

---

## 👨‍💻 Author

**Gaurang Chaturvedi**
AI & Data Science | Computer Vision | Deep Learning
GitHub: [https://github.com/Gaurang004](https://github.com/Gaurang004)


