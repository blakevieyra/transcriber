# Audio Transcription Tool

## Overview
This project provides a graphical user interface (GUI) for transcribing audio files using OpenAI's Whisper model. Users can easily select an audio file, initiate transcription, and view/save the resulting text. The tool is built with Python and utilizes the `tkinter` library for the GUI and `torch` for model computations.

## Features
- **Whisper Model Integration:** Supports transcription of `.m4a`, `.mp3`, and `.wav` audio formats using the Whisper model.
- **GPU Support:** Automatically detects CUDA availability and utilizes the GPU for faster processing if available.
- **Progress Bar:** Displays an indeterminate progress bar during transcription.
- **Timer:** Measures and displays the time taken for each transcription.
- **Threading:** Ensures a responsive interface by running transcription in a separate thread.
- **File Selection:** Allows users to select audio files through a file dialog.
- **Text Output:** Displays the transcription in a text box and saves it to a file.

## Requirements
- Python 3.7 or higher
- Libraries:
  - `torch`
  - `whisper`
  - `tkinter` (comes pre-installed with Python)

## Installation
1. Install Python if not already installed. [Download Python](https://www.python.org/downloads/)
2. Install the required Python libraries:
   ```bash
   pip install torch whisper
   ```

## Usage
1. Run the script:
   ```bash
   python <script_name>.py
   ```
2. The GUI window will appear.
3. Click the **Select Audio File and Transcribe** button to open a file dialog.
4. Select an audio file in one of the supported formats.
5. The transcription process will start. A progress bar will indicate activity, and the estimated time will be displayed upon completion.
6. View the transcription in the text box or open the saved `transcription_output.txt` file.

## Code Details
### Main Components
1. **File Selection:**
   ```python
   audio_file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.m4a;*.mp3;*.wav")])
   ```
2. **Whisper Model Loading:**
   ```python
   model = whisper.load_model("small")
   ```
3. **Transcription:**
   ```python
   result = model.transcribe(audio_file_path)
   transcription_output = result["text"]
   ```
4. **Progress Bar:**
   ```python
   progress_bar.start()  # Start animation
   progress_bar.stop()   # Stop animation
   ```
5. **Threading:**
   ```python
   threading.Thread(target=run_transcription).start()
   ```

### File Output
Transcriptions are saved to a file named `transcription_output.txt` in the current directory.

## Notes
- Ensure you have the necessary compute resources for the Whisper model, especially for large audio files.
- For CUDA support, ensure the correct version of PyTorch is installed with GPU capabilities.

## Troubleshooting
- **Whisper model not found:** Ensure the `whisper` library is correctly installed.
- **Audio file format issues:** Verify the selected audio file is in `.m4a`, `.mp3`, or `.wav` format.
- **Slow performance:** Use a GPU-enabled machine for faster transcription or reduce audio file size.

## Acknowledgments
- OpenAI for the Whisper model.
- Python community for libraries like `torch` and `tkinter`.

