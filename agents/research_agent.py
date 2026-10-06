from langchain_groq import ChatGroq
from typing import Dict, List
from langchain_core.documents import Document
import os

class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with Groq LLM.
        """
        print("Initializing ResearchAgent with Groq LLM...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        self.llm = ChatGroq(
            model_name="openai/gpt-oss-20b",
            temperature=0.1,
            max_tokens=512,  # Limit for faster responses
            groq_api_key=api_key,
        )
        print("LLM initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str, previous_answer: str = "", verification_feedback: str = "") -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        feedback_section = ""
        if verification_feedback:
            feedback_section = f"""
        **Previous Attempt:**
        {previous_answer}

        **Feedback / Unresolved Issues from Previous Attempt:**
        {verification_feedback}

        **Important Fix Instructions:**
        Please correct the previous answer based on the feedback above. Ensure all claims are strictly supported by the available information below.
        """

        prompt = f"""
        You are a helpful AI assistant that answers questions based on available information.
    
        **Instructions:**
        - Answer the question below using only the information provided.
        - If the information doesn't contain the answer, say: "Sorry, I don't have any information about your question."
        - Be clear, concise, and factual.
        - Never mention "context", "documents", "provided information", or similar phrases.
        - Just give the answer naturally or say you don't know.
        {feedback_section}
        **Question:** {question}
        
        **Available information:**
        {context}
    
        **Answer:**
        """
        return prompt

    def generate(self, question: str, documents: List[Document], previous_answer: str = "", verification_feedback: str = "") -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        print(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        if not documents:
            print("No documents provided to generate an answer.")
            return {
                "draft_answer": "Sorry, I don't have any information about your question.",
                "context_used": ""
            }

        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM
        prompt = self.generate_prompt(question, context, previous_answer=previous_answer, verification_feedback=verification_feedback)
        print("Prompt created for the LLM.")

        # Call the LLM to generate the answer
        try:
            print("Sending prompt to the model...")
            response = self.llm.invoke(prompt)
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            return {
                "draft_answer": "Sorry, I encountered an error while generating the answer.",
                "context_used": context
            }

        # Extract and process the LLM's response
        draft_answer = self.sanitize_response(response.content) if response.content else "I cannot answer this question."

        print(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }