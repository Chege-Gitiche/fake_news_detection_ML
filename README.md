# 📰 Fake News Detection Web App (RAG + GPT-4o-mini)
[![MIT License][mit-shield]][mit]

This is a Flask-based web application that detects whether a news article is **real or fake** using a combination of:

- ✅ Semantic Search with Sentence Transformers + FAISS  
- ✅ Retrieval-Augmented Generation (RAG)  
- ✅ OpenAI's GPT-4o-mini model  
- ✅ Live article scraping from the web

Users simply **paste a news article URL** into a web form and receive an AI-generated classification and reasoning.

---

## 🚀 How It Works

1. Embeds thousands of real/fake news samples using `sentence-transformers`  
2. Uses **FAISS** to retrieve top 5 similar articles to the new article  
3. Feeds the article and retrieved context into `gpt-4o-mini` using LangChain  
4. Displays the model’s prediction + justification

---

## 📁 Project Structure

```
.
├── app.py # Flask backend
├── templates/
│ └── index.html # Web UI
├── Fake.csv # Dataset: Fake news samples
├── True.csv # Dataset: Real news samples
├── news_faiss.index # Saved FAISS index (auto-generated)
├── news_metadata.pkl # Saved metadata (auto-generated)
└── .env # Contains your OpenAI API key

```

## 🚀 Project Setup

---

### 🧩 Dependencies Needed

The system requires the following libraries, packages, and frameworks:

| Tool / Library | Purpose |
|----------------|---------|
| [Python](https://www.python.org/) | Programming language |
| [Flask](https://flask.palletsprojects.com/) | Lightweight web framework |
| [HTML](https://developer.mozilla.org/en-US/docs/Web/HTML) | Markup language for the web UI |
| [Bootstrap](https://getbootstrap.com/) | Frontend styling and responsiveness |
| [Sentence-Transformers](https://www.sbert.net/) | Generating semantic embeddings |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search engine |
| [LangChain](https://www.langchain.com/) | Framework for RAG pipelines with LLMs |
| [OpenAI API](https://platform.openai.com/) | GPT-4o-mini for prediction and reasoning |
| [Newspaper3k](https://newspaper.readthedocs.io/) | Article scraping from web URLs |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |

---

### 📦 Packages to Download

Make sure the following are installed on your system:

- [Python](https://www.python.org/downloads/)
- [FAISS CPU](https://github.com/facebookresearch/faiss) *(optional but recommended)*
- [Git](https://git-scm.com/downloads)

---

### 🛠 Installation Steps

---

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

    Note:You can clone the repo in the directory of your choosing
#### 2. **Create and Activate a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

    Note:You must be within the directory of the project
#### 3. **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

    
    Note:If you do not have the requirements.txt file you can use the following command
    
    ```bash
    pip install flask python-dotenv faiss-cpu pandas tqdm \
    sentence-transformers newspaper3k   langchain langchain-openai openai
    ```
#### 4. **Download datasets from the kaggle website**
Download the zip folder from the website I send the link of then extract the .csv files and finally add the files after extraction to the working repository.

#### 5. **Create the .env File for important details**

   Create a .env file in the root directory of the project. This file will contain your confidential information. 

   ```
   OPENAI_API_KEY=Put_your_openai_api key here
   ```


## 📘 Usage Instructions

---

### ▶️ How to Run the System

Use the following command to start the Flask web server:

```bash
python app.py
```

Once the server starts, open your browser and navigate to

```bash
http://127.0.0.1:5000/
```
### 📥 Expected Input Format

- Article URL Input <br>
The user pastes a valid news article URL (e.g. from BBC, Reuters, CNN) into the form.

### 📤 Output Format

- Prediction Output <br>
The system returns a detailed natural language explanation of whether the article is likely to be Real or Fake, with justification.
- Token Usage Stats <br>
The app displays token usage info include Prompt Tokens, Completion Tokens and Total Tokens used during prediction
   
## 🗂️ Key Files

---

| File | Purpose |
|------|---------|
| `app.py` | Main Flask application file. Handles routing, embedding logic, article scraping, RAG invocation, and response rendering. |
| `templates/index.html` | The HTML user interface for submitting article URLs and displaying predictions. Styled using Bootstrap. |
| `Fake.csv` / `True.csv` | Datasets of labeled fake and real news articles. Used to build the semantic search index. |
| `news_faiss.index` | Saved FAISS vector index of all embedded article vectors. Automatically generated if not found. |
| `news_metadata.pkl` | Metadata (title, text, label) of all embedded articles for retrieval context. Saved as a pickle file. |
| `.env` | Stores sensitive variables like your `OPENAI_API_KEY`. Keep this file secret and never commit it to version control. |

## Contact Information

For any inquiries, please contact us at:
https://github.com/Chege-Gitiche
     