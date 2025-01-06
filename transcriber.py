import whisper
import torch
from tkinter import Tk, Label, Button, Text, filedialog, END, ttk
import time
import threading

def transcribe_audio():
    def run_transcription():
        audio_file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.m4a;*.mp3;*.wav")])
        if audio_file_path:
            status_label.config(text=f"Transcribing: {audio_file_path}")
            progress_bar.start()  
            root.update()

            global model
            if model is None:
                status_label.config(text="Loading Whisper model...")
                root.update()
                model = whisper.load_model("small")

            start_time = time.time()

            result = model.transcribe(audio_file_path)
            transcription_output = result["text"]

            end_time = time.time()
            elapsed_time = end_time - start_time

            transcription_box.delete(1.0, END)
            transcription_box.insert(END, transcription_output)

            output_file = "transcription_output.txt"
            with open(output_file, "w") as file:
                file.write(transcription_output)

            progress_bar.stop()  
            status_label.config(text=f"Transcription completed in {elapsed_time:.2f} seconds! Saved to {output_file}")
        else:
            progress_bar.stop() 
            status_label.config(text="No file selected.")

    threading.Thread(target=run_transcription).start()

cuda_available = torch.cuda.is_available()
if cuda_available:
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Using CPU.")

model = None

root = Tk()
root.title("Audio Transcription Tool")

Label(root, text="Audio Transcription Tool", font=("Helvetica", 16)).pack(pady=10)
Button(root, text="Select Audio File and Transcribe", command=transcribe_audio, width=30).pack(pady=10)
status_label = Label(root, text="", font=("Helvetica", 12))
status_label.pack(pady=5)
progress_bar = ttk.Progressbar(root, orient="horizontal", mode="indeterminate", length=400)
progress_bar.pack(pady=5)
Label(root, text="Transcription Output:", font=("Helvetica", 14)).pack(pady=5)
transcription_box = Text(root, wrap="word", width=60, height=20)
transcription_box.pack(pady=10)

root.mainloop()
