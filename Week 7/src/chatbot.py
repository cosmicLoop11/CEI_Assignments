import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = cohere.Client(api_key=cohere_api_key)

    def respond(self, user_message: str):
        """Interprets incoming queries and maps documents safely using the vectorstore."""
        retrieved_docs = []
        
        try:
            # Step 1: Use the query to pull matching documents from your vector database
            retrieved_docs = self.vectorstore.retrieve(user_message)
        except Exception as e:
            print(f"Retrieval warning: {str(e)}")
            retrieved_docs = []
            
        # Step 2: Stream grounded contextual responses using a current active model
        if retrieved_docs:
            stream = self.co.chat_stream(
                message=user_message,
                model="command-r-08-2024",
                documents=retrieved_docs,
                conversation_id=self.conversation_id,
            )
        else:
            stream = self.co.chat_stream(
                message=user_message,
                model="command-r-08-2024",
                conversation_id=self.conversation_id,
            )
            
        return stream, retrieved_docs