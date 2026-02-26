import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file.")


client = Groq(api_key=api_key)


def audit_transcript(transcript_text):
    prompt = f"""
You are an AI Customer Support Quality Auditor.

Analyze the transcript and provide structured evaluation.

Return output STRICTLY in the format below:

Empathy Level: <High/Medium/Low>
Empathy Reason: <One-line explanation>

Compliance Status: <Compliant/Non-Compliant>
Compliance Reason: <One-line explanation>

Quality Score: <Score out of 10>
Score Reason: <One-line explanation>

Customer Behavior Insight: <One-line explanation of why the customer is behaving this way>

Transcript:
{transcript_text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    sample_transcript = """
Customer: I am extremely disappointed with the delay in delivery.
Agent: I sincerely apologize for the inconvenience caused. Let me check your order and resolve this immediately.
"""

    print("\n🔍 Auditing Transcript...\n")

    result = audit_transcript(sample_transcript)

    print("✅ Audit Result:\n")
    print(result)