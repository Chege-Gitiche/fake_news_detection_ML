from flask import Flask, render_template, request
from dotenv import load_dotenv
import os, pickle, faiss, pandas as pd
from tqdm import tqdm
from newspaper import Article, Config
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
tqdm.pandas()

# Initialize Flask app
app = Flask(__name__)

# Paths
FAISS_PATH = "news_faiss.index"
METADATA_PATH = "news_metadata.pkl"

# Load or create embeddings
def prepare_embeddings():
    global model, index, metadata

    print("📦 Checking embeddings and metadata...")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    if not os.path.exists(FAISS_PATH) or not os.path.exists(METADATA_PATH):
        print("🔧 Building FAISS index and metadata from CSVs...")

        fake_df = pd.read_csv('Fake.csv').head(10000)
        real_df = pd.read_csv('True.csv').head(10000)
        fake_df['label'] = 0
        real_df['label'] = 1
        df = pd.concat([fake_df, real_df])
        df['content'] = df['title'] + ". " + df['text']

        embeddings = model.encode(df['content'].progress_apply(str).tolist(), convert_to_numpy=True)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, FAISS_PATH)

        metadata = df[['title', 'text', 'label']].to_dict('records')
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
        print("✅ Embeddings and metadata created.")
    else:
        print("📁 Loading existing FAISS index and metadata...")
        index = faiss.read_index(FAISS_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

# Initialize once on app startup
prepare_embeddings()

# LangChain setup
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
prompt_template = PromptTemplate(
    input_variables=["article", "context"],
    template="""
Based on the following known articles:
{context}

And the new article:
{article}

Is this new article likely to be real or fake? Justify your answer with reasoning.
"""
)
rag_chain = prompt_template | llm

# Web routes
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    tokens = {}

    if request.method == "POST":
        url = request.form["url"].strip()
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        config = Config()
        config.browser_user_agent = user_agent

        try:
            print("🔗 Downloading and parsing article...")
            article = Article(url, config=config)
            article.download()
            article.parse()
            scraped_text = article.title + ". " + article.text
            print("✅ Article scraped successfully.")

            print("📌 Embedding query article...")
            query_embedding = model.encode([scraped_text])
            print("✅ Embedding complete.")

            print("🔍 Retrieving similar articles from FAISS index...")
            D, I = index.search(query_embedding, k=5)
            similar_articles = [metadata[i] for i in I[0]]
            print("✅ Retrieved similar articles.")

            print("🧠 Running LLM RAG prediction...")
            context = "\n---\n".join([
                f"Title: {a['title']}\nText: {a['text']}\nLabel: {'Fake' if a['label'] == 0 else 'Real'}"
                for a in similar_articles
            ])
            output = rag_chain.invoke({"article": scraped_text, "context": context})
            prediction = output.content.strip()
            print("✅ RAG Prediction complete.")

            if hasattr(output, "response_metadata") and "token_usage" in output.response_metadata:
                usage = output.response_metadata["token_usage"]
                tokens = {
                    "prompt": usage.get("prompt_tokens", "N/A"),
                    "completion": usage.get("completion_tokens", "N/A"),
                    "total": usage.get("total_tokens", "N/A")
                }
                print("📊 Token usage:", tokens)

        except Exception as e:
            print("❌ Error during article analysis:", e)
            prediction = f"❌ Error: {e}"

    return render_template("index.html", prediction=prediction, tokens=tokens)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
