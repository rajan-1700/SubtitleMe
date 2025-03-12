import whisper

model = whisper.load_model("base")

def transcribe_audio(audio_file):
    result = model.transcribe(audio_file)
    segments = result["segments"]
    
    transcription = []
    for segment in segments:
        transcription.append({
            "text": segment["text"],
            "timestamp": [segment["start"], segment["end"]]
        })

    return transcription
