from rag.retriever import get_retriever
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def rag_audit(transcript):

    retriever = get_retriever()
    docs = retriever.invoke(transcript)

    policy_context = "\n".join([doc.page_content for doc in docs])

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
You are a Customer Support Quality Auditor.

Company Policies:
{policy_context}

Customer Call Transcript:
{transcript}

Evaluate:
1. Compliance with policies
2. Empathy level
3. Professionalism
4. Customer behavior
5. Final quality score (0-100)

Give structured output.
"""

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content