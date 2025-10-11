import whisper
import tempfile
import os
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Load Whisper model once globally
model_path = os.path.expanduser("~/.cache/whisper/small.en.pt")
model = whisper.load_model(model_path)

def transcribe_audio(audio_input):
    """
    Transcribe audio from either:
    - Streamlit UploadedFile
    - File path (str)
    """
    # Determine if input is UploadedFile or file path
    if isinstance(audio_input, UploadedFile):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_file.write(audio_input.read())
            temp_path = temp_file.name
    elif isinstance(audio_input, str) and os.path.exists(audio_input):
        temp_path = audio_input
    else:
        raise ValueError("Invalid input for audio transcription")

    # Transcribe audio
    result = model.transcribe(temp_path)

    # Remove temp file if it was created from UploadedFile
    if isinstance(audio_input, UploadedFile):
        os.unlink(temp_path)

    return result["text"]
