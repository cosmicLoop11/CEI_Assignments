import streamlit as st
import os
from dotenv import load_dotenv
from vectorstore import VectorStore
from chatbot import Chatbot

# Load environment keys if present locally
load_dotenv()

def main():
    st.set_page_config(page_title="RAG Document QA Bot", page_icon="🤖", layout="wide")
    st.title("Document QA Bot 🤖 (Advanced RAG Pipeline)")
    st.write("Upload custom text documents to ground model generations on context-verified facts.")

    # Initialize session state tracking
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = None

    # Sidebar settings
    with st.sidebar:
        st.header("Authentication Configuration 🔑")
        cohere_key = st.text_input("Cohere API Key", type="password", value=os.getenv("COHERE_API_KEY", ""))
        pinecone_key = st.text_input("Pinecone API Key", type="password", value=os.getenv("PINECONE_API_KEY", ""))
        
        st.markdown("---")
        st.markdown("### System Metrics Validation Logs")
        if st.session_state["vectorstore"]:
            st.success("✅ Knowledge Store Loaded")
            st.info(f"Chunks Count: {len(st.session_state['vectorstore'].chunks)}")
            st.info("Embedding Model: embed-english-v3.0")
            st.info("Vector DB Store: Pinecone Serverless")
            st.info("Generation Model: command-r")
        else:
            st.warning("⚠️ Pending Document Ingestion")

    # Document upload block
    uploaded_file = st.file_uploader("Upload an informational document (PDF)", type="pdf")
    
    if uploaded_file:
        # Avoid reloading and parsing vector index repeatedly if already handled
        if st.session_state["vectorstore"] is None:
            if not cohere_key or not pinecone_key:
                st.error("Please supply valid configuration access tokens in the sidebar first!")
                return
            
            with st.spinner("Executing Document Ingestion, Chunking & Vector Store Indexing..."):
                temp_pdf_path = "uploaded_document.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(uploaded_file.read())
                
                try:
                    # Ingest and map embeddings directly
                    vs = VectorStore(temp_pdf_path, cohere_key, pinecone_key)
                    st.session_state["vectorstore"] = vs
                    st.session_state["chatbot"] = Chatbot(vs, cohere_key)
                    st.toast("Document indexed successfully!", icon="🚀")
                except Exception as e:
                    st.error(f"Initialization Failed: {str(e)}")
                    return

    st.markdown("---")
    
    # UI Layout split for Chat history alongside retrieved chunks for verification
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Chat Assistant")
        user_query = st.text_input("Ask a question based on your document content:", key="query_input")
        submit_btn = st.button("Submit Query")

        if submit_btn and user_query:
            if not st.session_state["chatbot"]:
                st.error("Please upload and process a document before submitting questions.")
            else:
                with st.spinner("Searching logs and generating answer..."):
                    # Process response pipeline stream
                    stream_response, context_used = st.session_state["chatbot"].respond(user_query)
                    
                    # Display stream chunk rendering real-time
                    response_placeholder = st.empty()
                    accumulated_response = ""
                    
                    for event in stream_response:
                        if event.event_type == "text-generation":
                            accumulated_response += event.text
                            response_placeholder.markdown(f"**Bot (Streaming):** {accumulated_response}")
                    
                    # Log conversational state history
                    st.session_state["chat_history"].append({
                        "query": user_query,
                        "bot_response": accumulated_response,
                        "context": context_used
                    })

        # Render complete conversation sequence
        st.markdown("#### Conversation History")
        for chat in reversed(st.session_state["chat_history"]):
            st.markdown(f"👤 **You:** {chat['query']}")
            st.markdown(f"🤖 **Bot:** {chat['bot_response']}")
            st.markdown("---")

    with col2:
        st.subheader("Retrieved Context Chunks")
        st.write("Verifiable source content mapped dynamically via Reranker:")
        if st.session_state["chat_history"]:
            latest_chat = st.session_state["chat_history"][-1]
            for idx, doc in enumerate(latest_chat["context"]):
                with st.expander(f"Context Block Rank #{idx+1}"):
                    st.write(doc["text"])
        else:
            st.caption("No queries run yet. Source document chunks will render here to display grounding metrics.")

if __name__ == "__main__":
    main()