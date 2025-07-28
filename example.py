from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
)

# 加载 Multi30k 数据（英文-德文）
raw_datasets = load_dataset("bentrevett/multi30k", split={"train": "train", "validation": "validation", "test": "test"})

# 使用 MarianMT 英德预训练分词器
model_checkpoint = "Helsinki-NLP/opus-mt-en-de"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# 分词函数
max_input_length = 128
max_target_length = 128

def preprocess(example):
    inputs = tokenizer(example['en'], max_length=max_input_length, truncation=True)
    targets = tokenizer(example['de'], max_length=max_target_length, truncation=True)
    inputs['labels'] = targets['input_ids']
    return inputs

tokenized_datasets = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets["train"].column_names)

# 数据整理器（自动 padding 和对齐）
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 训练配置
training_args = Seq2SeqTrainingArguments(
    output_dir="./multi30k_transformer",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=False,  # 如果你有支持的 GPU
)

# BLEU 评估函数
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 对标签进行解码和清理
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]  # sacrebleu expects list of references

    return metric.compute(predictions=decoded_preds, references=decoded_labels)

# 训练器
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 开始训练
trainer.train()

# 在测试集评估
results = trainer.evaluate(tokenized_datasets["test"])
print(results)