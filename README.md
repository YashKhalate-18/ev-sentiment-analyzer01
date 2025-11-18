# ev-sentiment-analyzer01

**Problem statement:** In the rapidly evolving electric vehicle (EV) industry, designers struggle to translate unstructured customer feedback from thousands of reviews into actionable design changes. Raw text opinions (e.g., "the front looks too boxy") are difficult to quantify and visualize, creating a gap between public sentiment and new concept designs.

---

### 🧠 Sentiment-Driven EV Generative Designer
### 📋 Project Overview

This repository contains an end-to-end AI pipeline that bridges the gap between customer feedback and creative design. It contains three main parts:

* **NLP Sentiment Analysis** – An Aspect-Based Sentiment Analysis (ABSA) model that reads customer reviews to extract specific EV features (e.g., "headlights," "design") and the sentiment (Positive/Negative) associated with them.
* **AI Prompt Engineering** – A logic-based module that automatically converts the structured sentiment data into a rich, descriptive text prompt for a generative AI.
* **Generative AI Design** – A text-to-image (Stable Diffusion) model that uses the generated prompt to create novel, photorealistic concept art of an EV that reflects the public's opinions.

The goal is to combine AI-powered language understanding with generative AI to create a feedback loop where customer desires directly influence new designs.

### 📂 Repository Structure

### ⚙️ Features
🔹 **NLP Aspect-Based Sentiment Analysis (ABSA)**
* Uses a pre-trained `pyabsa` (Transformer) model.
* Identifies specific topics in reviews (e.g., "wheels," "interior") beyond just a general review score.
* Classifies sentiment (Positive/Negative) for each specific topic.

🔹 **Generative AI Pipeline**
* Uses `diffusers` (Stable Diffusion) to generate 4K, photorealistic images.
* Automatically translates NLP insights into creative prompts.

🔹 **Interactive Web App**
* Built with Streamlit for a simple, user-friendly interface.
* Allows a user to **upload a CSV** of reviews.
* User specifies the review column to analyze.
* Displays the sentiment summary, the final prompt, and the resulting AI-generated image.

### 🧰 Technologies Used
| Area | Tools / Libraries |
| :--- | :--- |
| **Programming Language** | `Python 3.x` |
| **Libraries** | `pandas`, `numpy` |
| **NLP / ML** | `pyabsa`, `transformers` |
| **Generative AI** | `diffusers` (Stable Diffusion), `torch` |
| **Web App / UI** | `streamlit` |
| **Environment** | `VS Code` |
| **Version Control** | `Git + GitHub` |


