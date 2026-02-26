import whisper
import os
import pandas as pd


model = whisper.load_model("base")

audio_folder = "Datasets"
output_data = []

for file in os.listdir(audio_folder):
    if file.endswith(".mp3"):
        file_path = os.path.join(audio_folder, file)
        
        print(f"Transcribing {file}...")
        result = model.transcribe(file_path)
        
        output_data.append({
            "file_name": file,
            "transcript": result["text"]
        })

df = pd.DataFrame(output_data)
df.to_csv("transcripts/transcripts.csv", index=False)

print("Transcription completed.")