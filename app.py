from flask import Flask, request, render_template, send_from_directory, send_file, jsonify
import os
import re
from extract_audio import extract_audio_from_video
from transcribe_audio import transcribe_audio

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
SUBTITLE_FOLDER = 'subtitles'  # Store VTT & TXT files here
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUBTITLE_FOLDER, exist_ok=True)

def clean_vtt(vtt_path):
    """
    Reads a VTT file and removes timestamps to return plain text.
    """
    if not os.path.exists(vtt_path):
        return "No transcript available."

    with open(vtt_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    clean_text = []
    for line in lines:
        # Remove timestamps and metadata
        if not re.match(r'\d+:\d+:\d+\.\d+ --> \d+:\d+:\d+\.\d+', line) and not line.strip().isdigit():
            clean_text.append(line.strip())

    return "\n".join(clean_text)

@app.route('/download_vtt/<filename>')
def download_vtt(filename):
    """ Serve the original VTT file for download from the subtitles folder """
    vtt_path = os.path.join(SUBTITLE_FOLDER, filename)
    if os.path.exists(vtt_path):
        return send_file(vtt_path, as_attachment=True)
    else:
        return "VTT file not found", 404

@app.route('/download_script/<filename>')
def download_script(filename):
    """ Serve the cleaned script (without timestamps) as a downloadable .txt file """
    vtt_path = os.path.join(SUBTITLE_FOLDER, filename)

    if not os.path.exists(vtt_path):
        return "VTT file not found", 404

    script_text = clean_vtt(vtt_path)
    script_filename = filename.rsplit('.', 1)[0] + "_clean.txt"
    script_path = os.path.join(SUBTITLE_FOLDER, script_filename)

    # ✅ Ensure the cleaned transcript is saved before downloading
    with open(script_path, "w", encoding="utf-8") as file:
        file.write(script_text)

    if os.path.exists(script_path):
        return send_file(script_path, as_attachment=True)
    else:
        return "Transcript file not found", 404

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Extract audio
        audio_file = extract_audio_from_video(file_path)
        if audio_file is None:
            return "Audio extraction failed", 500

        # Transcribe audio
        transcription_chunks = transcribe_audio(audio_file)
        if transcription_chunks is None:
            return "Transcription failed", 500

        # Generate VTT file
        subtitle_filename = file.filename.rsplit('.', 1)[0] + ".vtt"
        subtitle_file = os.path.join(SUBTITLE_FOLDER, subtitle_filename)
        generate_vtt_file(transcription_chunks, subtitle_file)

        return jsonify({
            "subtitle_url": f"/subtitles/{subtitle_filename}",
            "download_vtt_url": f"/download_vtt/{subtitle_filename}",
            "download_script_url": f"/download_script/{subtitle_filename}"
        })

@app.route('/subtitles/<filename>')
def get_subtitles(filename):
    return send_from_directory(SUBTITLE_FOLDER, filename)

def generate_vtt_file(transcription_chunks, subtitle_file):
    with open(subtitle_file, 'w', encoding='utf-8') as f:
        f.write("WEBVTT\n\n")
        for idx, chunk in enumerate(transcription_chunks):
            start_time = format_time(chunk["timestamp"][0])
            end_time = format_time(chunk["timestamp"][1])
            f.write(f"{idx+1}\n{start_time} --> {end_time}\n{chunk['text']}\n\n")

def format_time(seconds):
    millis = int((seconds % 1) * 1000)
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{millis:03}"

if __name__ == '__main__':
    app.run(debug=True)

