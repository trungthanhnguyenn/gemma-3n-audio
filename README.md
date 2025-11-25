# Gemma 3N Audio Finetuning

This project demonstrates how to fine-tune the Gemma 3N model for audio transcription tasks. It includes scripts for both training and inference.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/trungthanhnguyenn/gemma-3n-audio.git
    cd gemma-3n-audio
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Create a `.env` file in the root directory and add your Hugging Face Hub token:
    ```
    HUGGINGFACE_HUB_TOKEN=your_hugging_face_token
    ```

## Dataset

The training script uses the `nguyendv02/ViMD_Dataset` dataset from the Hugging Face Hub. Specifically, it filters for samples from the "Central" region.

## Training

The `train.py` script handles the fine-tuning process.

### Key Steps in `train.py`:

1.  **Load Pretrained Model:** Loads the `unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit` model using `FastModel` from the `unsloth` library.
2.  **Load and Preprocess Dataset:** Loads the `nguyendv02/ViMD_Dataset`, filters for the "Central" region, and formats it to match the Gemma 3N multimodal chat format.
3.  **PEFT Configuration:** Sets up the model for Parameter-Efficient Fine-Tuning (PEFT) using LoRA.
4.  **Training:** Uses the `SFTTrainer` from the `trl` library to fine-tune the model on the preprocessed dataset.
5.  **Save and Push to Hub:**
    *   Saves the LoRA adapter to the `gemma-3n-audio-finetuned` directory.
    *   Pushes the LoRA adapter to the Hugging Face Hub.
    *   Merges the adapter with the base model and pushes the merged model to the Hugging Face Hub.

### How to Run `train.py`:

```bash
python train.py
```

## Inference

The `inference.py` script demonstrates how to use the fine-tuned model for audio transcription.

### Key Features of `inference.py`:

*   **`load_audio(path)`:** Loads an audio file, resamples it to 16kHz, converts it to mono, and normalizes the audio array.
*   **`transcribe_from_array(audio_array)`:** Takes an audio array (e.g., from the dataset) and returns the transcription.
*   **`transcribe_from_file(audio_path)`:** A convenience wrapper that takes a file path, loads the audio using `load_audio`, and then transcribes it using `transcribe_from_array`.

### How to Run `inference.py`:

The script, when run directly, will:
1.  Load a sample from the "Central" region of the `nguyendv02/ViMD_Dataset`.
2.  Transcribe the audio from that sample.
3.  Print the ground truth and the predicted transcription.
4.  If `examples/test.wav` exists, it will also transcribe that file.

To run the script:

```bash
python inference.py
```

## Evaluation 

The evaluation script for WER/CER will update soon