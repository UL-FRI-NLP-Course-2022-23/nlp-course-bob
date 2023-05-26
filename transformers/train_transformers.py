from transformers import get_linear_schedule_with_warmup
import torch
import os
from torch.optim import AdamW
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils import tensor_datasets, convert_to_input, load_data, load_models
import numpy as np

MAX_LENGTH = 150
BATCH_SIZE = 16

def main():
    absolute_path = os.path.dirname(os.path.dirname(__file__))
    # Load data
    data = pd.read_csv(absolute_path + '/data/bigger dataset/paraphrases_30k_filtered.csv')
    xy_train, xy_val, xy_test = preprocess_data(data)
    
    # Load models
    print(f'Loading model...')
    tokenizer, model= load_models(absolute_path + "/models/t5_small")
    print('Model loaded!')

    device = "cpu"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\nFound GPU device: {torch.cuda.get_device_name(i)}")
        device = "cuda"
    
    device = torch.device("cuda")
    model.cuda()
    
    ##########################################
    #                Train

    x_train,  y_train, masks_train = convert_to_input(xy_train, tokenizer)
    x_val,  y_val, masks_val = convert_to_input(xy_val, tokenizer)
    x_test,  y_test, masks_test = convert_to_input(xy_test, tokenizer)
    
    train_dataloader = tensor_datasets(x_train, y_train, masks_train, BATCH_SIZE)
    val_dataloader = tensor_datasets(x_val, y_val, masks_val, BATCH_SIZE)
    test_dataloader = tensor_datasets(x_test, y_test, masks_test, BATCH_SIZE)
    
    print(f'Len of actual training data: {len(x_train)}')

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=2e-5,
                      eps=1e-8)
    
    loss_values, validation_loss_values = [], []
    epochs = 20
    max_grad_norm = 1.0
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps)

    for epoch_id in range(epochs):
        model.cuda()
        print(f'Epoch {epoch_id+1}')
        model.train()
        total_loss = 0
        
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            b_input_ids, b_input_mask, b_output_ids = tuple(t.to(device) for t in batch)
            
            model.zero_grad()
            outputs = model(input_ids = b_input_ids,
                            attention_mask = b_input_mask,
                            labels = b_output_ids)
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
                    
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        
        ##############################################
        #               Validation
        
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask, 
                                labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        
    torch.save(model, 'models/t5_final.pt')


def preprocess_data(data):
    # Split data
    x_train, x_test = train_test_split(data, test_size=0.20, shuffle=False, random_state = 10)
    x_val, x_test = train_test_split(x_test, test_size=0.10, shuffle=False, random_state = 10)
    
    return x_train, x_val, x_test


if __name__ == '__main__':
    main()
    

    