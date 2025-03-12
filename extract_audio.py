from processing import subprocess, os

def extract_audio_from_video(video_file):
    # Check if the input video file exists
    if not os.path.isfile(video_file):
        print(f"File '{video_file}' does not exist.")
        return None

    # Define the output audio file name (WAV format)
    audio_file = os.path.splitext(video_file)[0] + "_audio.wav"

    # Construct the FFmpeg command
    command = [
        'ffmpeg',
        '-i', video_file,
        '-q:a', '0',         # Set audio quality
        '-map', 'a',         # Select the audio stream
        audio_file           # Output audio file
    ]

    try:
        # Run the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Audio extracted successfully to '{audio_file}'")
        return audio_file
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None
