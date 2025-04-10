__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Gemini Embedding wrapper compatible with LangChain
class GeminiEmbeddingFunction(Embeddings):
    def __init__(self, document_mode=True):
        self.document_mode = document_mode

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            response = genai.embed_content(
                model="models/text-embedding-004", content=text, task_type="retrieval_document"
            )
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text):
        response = genai.embed_content(
            model="models/text-embedding-004", content=text, task_type="retrieval_query"
        )
        return response["embedding"]

# Constants
DB_NAME = "program_recommendation_db"
PERSIST_DIR = "chroma_persistent_storage"
COLLECTION_NAME = DB_NAME

# Initialize embedding model
embedding_model = GeminiEmbeddingFunction(document_mode=True)

# Load data from Excel
uploaded_file = "program_categorization.xlsx"

st.set_page_config(page_title="Health Program Recommender", layout="wide")
st.title("🏥 Personalized Health Program Recommender")


if os.path.exists(uploaded_file):
    df = pd.read_excel(uploaded_file)

    vectorstore_exists = os.path.exists(f"{PERSIST_DIR}/index")

    if not vectorstore_exists:
        documents = []
        for _, row in df.iterrows():
            content = f"""
            Program Name: {row['Program Name']}
            Duration: {row['Duration']}
            Medical Tests: {row['Medical tests '] }
            Caters To: {row['Caters to']}
            Emotional Counselors Support: {row['Emotional counselors support']}
            Group Yoga: {row['Group Yoga']}
            Life Coach Call: {row['Life Coach call']}
            Benefits: {row['Benefits ']}
            Conditions Covered: {row['List of Medical conditions covered']}
            Pricing: {row['Pricing and Packages ']}
            Testimonials: {row['Testimonials ']}
            """
            documents.append(Document(page_content=content.strip()))

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
        vectorstore.persist()
    else:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )

    # Input UI
    user_query = st.text_area("✍️ Describe your (or someone else's) current health situation", height=200)

    if st.button("🔍 Get Recommendation"):
        if user_query.strip() == "":
            st.warning("Please enter a health query.")
        else:
            embedding_model.document_mode = False
            results = vectorstore.similarity_search(user_query, k=3)

            if results:
                context = "\n\n".join(doc.page_content for doc in results)
                prompt = f"""
                You are Luke Coutinho, a holistic wellness coach known for personalized, mindful advice.

                Using the context provided below—which contains various wellness programs, Based on the following programs and the user's query, recommend the **single most suitable program** in a short, friendly 5 sentences. Keep it simple, clear, and compassionate—like something Luke would say in a quick voice note.

                Avoid jargon—speak in a friendly, compassionate, and reassuring tone.

                Context:
                {context}

                User Query:
                {user_query}

                Luke's Personalized Recommendation:
                """

                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                response = model.generate_content(prompt)

                st.subheader("🧘‍♂️ Luke's Personalized Recommendation")
                # st.markdown(response.text.strip())
                with st.container():
                    st.markdown(
                        f"""
                        <div style="background-color:#f1f8f6; padding:20px; border-radius:12px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
                            <div style="display: flex; align-items: center; gap: 12px;">
                                <img src="https://i.imgur.com/1X4K3nL.png" alt="Luke" width="48" style="border-radius: 50%;" />
                                <div>
                                    <strong style="font-size: 1.1rem;">Luke Coutinho</strong><br>
                                    <span style="font-size: 0.9rem; color: #555;">Holistic Wellness Coach</span>
                                </div>
                            </div>
                            <div style="margin-top: 16px; font-size: 1rem; color: #222;">
                                {response.text.strip()}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error("⚠️ No relevant programs found.")
else:
    st.error("❌ Excel file not found. Please check the file path.")
