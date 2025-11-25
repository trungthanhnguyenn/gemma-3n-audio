from unsloth import FastModel
import torch
from huggingface_hub import snapshot_download
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv
import os

load_dotenv()

# =========================
# 1. Load Pretrained Model
# =========================
model, processor = FastModel.from_pretrained(
    model_name="unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    dtype=None,
    max_seq_length=1024,
    load_in_4bit=True,
    full_finetuning=False,
)

# =========================
# 2. Dataset
# =========================
dataset = load_dataset("nguyendv02/ViMD_Dataset", split="train")
central = dataset.filter(lambda x: x["region"] == "Central")

# =========================
# 3. Preprocess function
# =========================
def format_intersection_data(samples: dict) -> dict:
    """
    Format ViMD dataset to match Gemma 3N multimodal chat format.
    """
    messages = []

    for audio, text in zip(samples["audio"], samples["text"]):
        message = [
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
                    {"type": "audio", "audio": audio["array"]},
                    {"type": "text", "text": "Please transcribe this audio."},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": text}
                ],
            },
        ]
        messages.append(message)

    return {
        "messages": messages,
        "audio": samples["audio"],
        # additional fields:
        # "region": samples["region"],
        # "province": samples["province"],
        # ...
    }


train_dataset = central.map(
    format_intersection_data,
    batched=True,
    batch_size=4,
    num_proc=4,
)

# =========================
# 4. Collate function
# =========================
def collate_fn(examples):
    """Collate for ViMD + Gemma 3N Audio."""
    texts = []
    audios = []

    for example in examples:
        text = processor.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        ).strip()
        texts.append(text)

        audios.append(example["audio"]["array"])

    batch = processor(
        text=texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=16000,
    )

    labels = batch["input_ids"].clone()

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    for attr in ["audio_token_id", "boi_token_id", "eoi_token_id", "image_token_id"]:
        token_id = getattr(processor.tokenizer, attr, None)
        if token_id is not None:
            labels[labels == token_id] = -100

    batch["labels"] = labels
    return batch


# =========================
# 5. LoRA / PEFT config
# =========================
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # Audio-related modules
        "post",
        "linear_start",
        "linear_end",
        "embedding_projection",
    ],
)

# =========================
# 6. Trainer setup
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    processing_class=processor.tokenizer,
    data_collator=collate_fn,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.1,
        num_train_epochs=1,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="", 
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=2,
        max_length=1024,
    ),
)

trainer_stats = trainer.train()

# =========================
# 7. Save & push to Hub
# =========================

# --- LORA adapter ---
model.save_pretrained("gemma-3n-audio-finetuned")
processor.save_pretrained("gemma-3n-audio-finetuned")

model.push_to_hub(
    "nguyenthanhtrung/gemma-3n-audio-it-unsloth-lora",
    token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
)
processor.push_to_hub(
    "nguyenthanhtrung/gemma-3n-audio-it-unsloth-lora",
    token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
)

# --- Full model (float16, merged) ---
model.save_pretrained_merged("gemma-3n-audio-finetuned", processor)
model.push_to_hub_merged(
    "nguyenthanhtrung/gemma-3n-audio-it-unsloth-full",
    token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
    processor=processor,
)
