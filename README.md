# 🛡️ SMS Spam Detector: AI‑Powered Message Guardian

> **Is that message a Gift 🎁 or a Trap 🪤?**
> An intelligent, AI‑powered system that instantly detects whether an SMS is **Ham (legitimate)** or **Spam (fraudulent/unwanted)** using Natural Language Processing.

---

## 🚀 Live Demo

🔗https://smsspamdetector-4771.streamlit.app/


---

## 📌 Project Overview

The **SMS Spam Detector** is a sophisticated text‑classification engine built to protect users from spam, phishing attempts, and fraudulent messages. It uses a **Multinomial Naive Bayes** model—an industry‑standard algorithm known for its efficiency and accuracy in NLP‑based document classification.

The application is wrapped in a **sleek Streamlit interface**, offering real‑time predictions, visual feedback, and an intuitive user experience.

---

## 🧠 How It Works

### 1️⃣ Text Vectorization (TF‑IDF)

Incoming messages are transformed into numerical vectors using **Term Frequency–Inverse Document Frequency (TF‑IDF)**. This process highlights important spam‑related keywords such as *winner*, *urgent*, *cash*, and *prize*, while filtering out common filler words.

### 2️⃣ Probabilistic Classification

The processed text is passed to a **Multinomial Naive Bayes classifier**, which applies **Bayes’ Theorem** to calculate the probability of a message being Spam or Ham. The trained model and vectorizer are stored in serialized files (`model.pkl`, `vectorizer.pkl`).

---

## ✨ Features

* ⚡ **Real‑time Detection** – Predictions in milliseconds
* 🎈 **Visual Feedback** – Emojis indicate results (Ham 🎈 | Spam ❄️)
* 🛑 **Phishing Protection** – Detects suspicious URLs and aggressive sales language
* 🎨 **Clean UI** – Centered layout with custom CSS styling
* 🤖 **AI‑Powered** – Reliable and scalable NLP‑based solution

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/sanchitsatpaise/SMS_spam_detector.git
cd SMS_spam_detector
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```plaintext
├── app.py              # Main Streamlit application
├── model.pkl           # Pre-trained Multinomial Naive Bayes model
├── vectorizer.pkl      # Pre-trained TF-IDF vectorizer
├── requirements.txt    # Required Python dependencies
└── README.md           # Project documentation
```

---

## 🧪 Example Tests

| Message Type | Input Sample                                                 | Prediction |
| ------------ | ------------------------------------------------------------ | ---------- |
| 🚨 Spam      | "WINNER! You have won a £1000 prize. Call 0905... to claim!" | SPAM       |
| ✅ Ham        | "Hey, are we still meeting for coffee at 4 PM?"              | HAM        |

---



## ❤️ Acknowledgements

Built with ❤️ using **Python, Streamlit, and Scikit‑learn**.

**Author:** *Sanchit Satpaise*
