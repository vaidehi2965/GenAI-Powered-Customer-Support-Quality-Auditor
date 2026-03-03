from transcribe import transcribe_audio
from rag.rag_auditor import rag_audit
from database import save_audit

def main():

    audio_path = "audio_file1.mp3"

    # Step 1: Convert audio to text
    transcript = transcribe_audio(audio_path)

    print("\nTranscript:\n", transcript)

    # Step 2: Run RAG audit
    result = rag_audit(transcript)

    print("\nFinal RAG Audit Report:\n")
    print(result)

    # Step 3: Save to database
    save_audit(transcript, result)


if __name__ == "__main__":
    main()