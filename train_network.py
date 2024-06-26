import random

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback, \
    TrainerCallback
from datasets import Dataset
import wandb


class PrintRandomSampleCallback(TrainerCallback):

    def __init__(self, _validation_dataset):
        self.validation_dataset = _validation_dataset

    def on_evaluate(self, args, state, control, **kwargs):
        try:
            # Get a random sample from the validation dataset
            random_idx = random.randint(0, len(self.validation_dataset) - 1)
            sample = self.validation_dataset[random_idx]
            input_decoded = sample['input_text']
            label_decoded = sample['label_text']
            input = tokenizer(input_decoded, return_tensors='pt', padding=True, truncation=True).to("cuda:0")

            # Tokenize and get the model prediction
            model.eval()
            with torch.no_grad():
                outputs = model.generate(input['input_ids'], attention_mask=input['attention_mask'],
                                         num_return_sequences=1, max_new_tokens=1,
                                         pad_token_id=tokenizer.pad_token_id)

            prediction_decoded = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True).replace(" ", "")

            # Print the results
            print(
                f"Random validation sample:\nI: {input_decoded}\nL: {label_decoded}\nR: {prediction_decoded}")
        except Exception as e:
            print(e)


def prepare_data(data, eos_token):
    """
    Changes data from format:
    1423
    to
    [A1][B4][A2][B3]
    expected by Neural network
    """
    players = ['A', 'B']
    player_index = 0

    inputs = list()
    labels = list()

    for line in data:
        out_line = []
        for char in line.split(" "):
            new_element = "[" + players[player_index] + char.replace("\n", "") + "]"
            if player_index == 1:
                inputs.append("".join(out_line))
            out_line.append(F"{new_element}")
            if player_index == 1:
                labels.append("".join(out_line) + eos_token)
            player_index = (player_index + 1) % 2
    return inputs, labels


if __name__ == "__main__":
    # Initialize Weights & Biases
    wandb.init(project="gpt2-agent")

    # Load saved training and validation data:
    training_data = list()
    with open("training_data.txt", "r") as file:
        training_data += file.readlines()

    validation_data = list()
    with open("validation_data.txt", "r") as file:
        validation_data += file.readlines()

    # Load tokenizer and model and tokenize data:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    train_inputs, train_labels = prepare_data(training_data, tokenizer.eos_token)
    val_inputs, val_labels = prepare_data(validation_data, tokenizer.eos_token)

    print(F"Example of data:\nInput:{train_inputs[1234]}\nLabel:{train_labels[1234]}")

    # Convert data to Huggingface's Dataset format
    train_dataset = Dataset.from_dict({"input_text": train_inputs, "label_text": train_labels})
    validation_dataset = Dataset.from_dict({"input_text": val_inputs, "label_text": val_labels})
    train_dataset = train_dataset.shuffle(seed=412)

    print("Number of games in the training dataset:", len(train_dataset))
    print("Number of games in the validation dataset:", len(validation_dataset))

    # Load GPT2 and add special tokens to tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Create a list of all possible tokens
    special_tokens = [f'[{letter}{number}]' for letter in 'AB' for number in range(7)]

    print("Added special tokens: ", special_tokens)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    tokenizer.save_pretrained('./tokenizer')
    model.resize_token_embeddings(len(tokenizer))

    max_length = max(len(tokenizer.encode(seq)) for seq in train_labels) + 10

    def tokenize_function(examples):
        inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=max_length)
        labels = tokenizer(examples['label_text'], padding="max_length", truncation=True, max_length=max_length)
        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
                'labels': labels['input_ids']}


    # Tokenize the data:
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True,
                                                remove_columns=["input_text", "label_text"])
    tokenized_val_dataset = validation_dataset.map(tokenize_function, batched=True,
                                                   remove_columns=["input_text", "label_text"])

    # Verify tokenized data
    print(tokenized_train_dataset[4])
    print(tokenizer.decode(tokenized_train_dataset[4]["input_ids"]))
    print(tokenizer.decode(tokenized_train_dataset[4]["labels"]))

    # Set up the Trainer
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=2_500,
        save_steps=2_500,
        save_total_limit=2,
        report_to="wandb",
        run_name="game_agent",
        lr_scheduler_type='cosine_with_restarts',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        callbacks=[PrintRandomSampleCallback(validation_dataset)]
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained('./fine-tuned-gpt2-agent')
    tokenizer.save_pretrained('./fine-tuned-gpt2-agent')

    # Finish the W&B run
    wandb.finish()