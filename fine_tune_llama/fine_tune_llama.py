from transformers import (
    AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
import torch
from datasets import Dataset
import json
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model, TaskType

# model configuration
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # smaller model for memory efficiency

# bitsandbytes configuration (4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# load model with quantization
model = LlamaForCausalLM.from_pretrained(
    model_name, 
    quantization_config=bnb_config, 
    device_map="auto", 
    torch_dtype=torch.bfloat16
)

# freeze base model layers enable only LoRA updates
for param in model.parameters():
    param.requires_grad = False  

# load dataset
with open("recipedataset.json") as f:
    data = json.load(f)

rows = data['rows']
data_list = [
    {"instruction": row["row"]["instruction"], "input": row["row"]["input"], "output": row["row"]["output"]} 
    for row in rows
]

# create dataset from list
dataset = Dataset.from_list(data_list)

# split dataset into training and validation
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# tokenization function
def tokenize_function(examples):
    inputs = [f"Instruction: {i}\nInput: {inp}\nResponse:" for i, inp in zip(examples['instruction'], examples['input'])]
    targets = [output for output in examples['output']]
    
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    # causal lm requires shifting labels
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# data collator for causal lm
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# lora configuration (peft)
lora_config = LoraConfig(
    r=32,  # increase rank for better fine-tuning
    lora_alpha=64,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)

# apply lora to model
model = get_peft_model(model, lora_config)

# training arguments play with these to fine tune
training_args = TrainingArguments(
    output_dir="./llama-recipe-finetune",
    per_device_train_batch_size=4,  # lower batch size for memory efficiency
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # larger effective batch size
    evaluation_strategy="steps",
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    save_total_limit=2,
    warmup_steps=200,
    learning_rate=1e-5,
    weight_decay=0.01,
    num_train_epochs=8,  # more epochs for better learning
    fp16=False,  # avoid conflicts with bfloat16 quantization
    group_by_length=True,  
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    data_collator=data_collator,
)


trainer.train()

model.save_pretrained("./llama-finetuned-recipe")
tokenizer.save_pretrained("./llama-finetuned-recipe")
