import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torchaudio
import numpy as np
from unsloth import FastModel
from datasets import load_dataset

# =======================
# 1. Load model & processor
# =======================
# MODEL_NAME = "gemma-3n-audio-finetuned"

# Load dataset
print("Loading ViMD dataset...")
dataset = load_dataset("nguyendv02/ViMD_Dataset", split="train")
central = dataset.filter(lambda x: x["region"] == "Central")
print(f"Central dataset size: {len(central)} samples")



model, processor = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)
model.to("cuda")


# =======================
# 2. Load audio
# =======================
def load_audio(path, target_sr=16000):
    """Load audio file, resample to 16kHz, normalize [-1, 1]."""
    waveform, sr = torchaudio.load(path)
    
    # Stereo -> mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    audio_array = waveform.squeeze().numpy().astype(np.float32)
    
    max_abs = np.abs(audio_array).max()
    if max_abs > 0:
        audio_array = audio_array / max_abs
    
    return audio_array, target_sr


# =======================
# 3. Transcription functions
# =======================
def transcribe_from_array(audio_array, target_sr=16000, max_new_tokens=256):
    """Transcribe from audio array (from dataset)"""

    if isinstance(audio_array, np.ndarray):
        max_abs = np.abs(audio_array).max()
        if max_abs > 1.0:
            audio_array = audio_array / max_abs
    
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an assistant that transcribes speech accurately.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        audio=[audio_array],
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_len = inputs["input_ids"].shape[1]
    gen_only_ids = output_ids[0, prompt_len:]

    transcript = processor.tokenizer.decode(
        gen_only_ids,
        skip_special_tokens=True,
    ).strip()

    return transcript


def transcribe_from_file(audio_path, max_new_tokens=256):
    """Transcribe from file audio"""
    audio_array, sr = load_audio(audio_path)
    return transcribe_from_array(audio_array, target_sr=sr, max_new_tokens=max_new_tokens)


# =======================
# 4. Main
# =======================
if __name__ == "__main__":
    if len(central) > 0:
        # Get first sample
        sample = central[0]
        print(f"Testing with dataset sample:")
        print(f"   Region: {sample['region']}")
        print(f"   Province: {sample.get('province', 'N/A')}")
        print(f"   Ground truth: {sample['text']}")
        
        # Transcribe
        audio_array = sample["audio"]["array"]
        sampling_rate = sample["audio"]["sampling_rate"]
        
        print(f"   Audio shape: {np.array(audio_array).shape}")
        print(f"   Sampling rate: {sampling_rate}")
        
        text = transcribe_from_array(audio_array, target_sr=sampling_rate)
        print("\n===== TRANSCRIPTION =====")
        print(text)
        
        print("\n===== COMPARISON =====")
        print(f"Ground truth: {sample['text']}")
        print(f"Prediction:   {text}")
    else:
        print("No samples found in central dataset!")
    
    # Also test with a local file
    audio_path = "examples/test.wav"
    if os.path.exists(audio_path):
        print(f"\nAlso testing with file: {audio_path}")
        text = transcribe_from_file(audio_path)
        print("===== FILE TRANSCRIPTION =====")
        print(text)
