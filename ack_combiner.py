from transformers import BertTokenizer, AutoTokenizer, PreTrainedEncoderDecoder
import load_data_test as load_data
import torch, sys
from tqdm import tqdm
import sys, time, datetime, random
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from SequenceCrossEntropyLoss import SequenceCrossEntropyLoss

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#load comments and labels from the input tsv
inputs, outputs = load_data.get_data(sys.argv[1])

print(outputs)

print(inputs)

# Load the BERT tokenizer.
print('Loading tokenizers...')
#tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
input_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", add_prefix_space=True)
output_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")

#encode inputs using BERT tokenizer
input_ids = []
output_ids = []

#input_ids = tokenizer.batch_encode_plus(inputs, max_length=1024, return_tensors='pt')
#output_ids = tokenizer.batch_encode_plus(outputs, max_length=1024, return_tensors='pt')

for in_data, output in zip(inputs, outputs):
     #print(in_data)
     #print(output)
     encoded_input = input_tokenizer.encode(in_data, add_special_tokens = True, max_length=256,pad_to_max_length=True)
     encoded_output = input_tokenizer.encode(output, add_special_tokens = True, max_length=256,pad_to_max_length=True)
     input_ids.append(encoded_input)
     output_ids.append(encoded_output)

#add padding to max len
def pad_input(input_ids):
    MAX_LEN = 128
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(input_tokenizer.pad_token, input_tokenizer.pad_token_id))
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long",
                          value=0, truncating="post", padding="post")
    print('\nDone.')
    return input_ids

input_ids = pad_input(input_ids)
output_ids = pad_input(output_ids)

#define attention masks: if 0 it's a PAD, set to 0; else set to 1
attention_masks = []

for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)

#train_test_val split
train_inputs, validation_inputs, train_outputs, validation_outputs = train_test_split(input_ids, output_ids,
                                                            random_state=2018, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, output_ids,
                                             random_state=2018, test_size=0.1)

# Convert all inputs and labels into torch tensors, the required datatype
# for our model.
train_inputs = torch.tensor(train_inputs)
print("input: ", train_inputs.shape)
validation_inputs = torch.tensor(validation_inputs)

train_outputs = torch.tensor(train_outputs)
print("output: ", train_outputs.shape)
validation_outputs = torch.tensor(validation_outputs)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# The DataLoader needs to know our batch size for training, so we specify it
# here.
# For fine-tuning BERT on a specific task, the authors recommend a batch size of
# 16 or 32.

batch_size = 4

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_outputs)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_outputs)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single
# linear classification layer on top.
model = PreTrainedEncoderDecoder.from_pretrained('bert-base-uncased', 'microsoft/DialoGPT-large')

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Tell pytorch to run this model on the GPU.
model.cuda()


# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 1e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )

# Number of training epochs (authors recommend between 2 and 4)
epochs = 12

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 600, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

prev_val_loss = 100000000.0

# For each epoch...
for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0
    total_val_loss = 0
#    loss_func = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss_func = SequenceCrossEntropyLoss()

    # Put the model into training mode. Don't be mislead--the call to
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # for name, param in model.module.named_parameters():
    #     if "layer_norm" not in name:
    #         param.requires_grad = False

    for name, param in model.module.named_parameters():
        if "model.decoder" not in name:
            param.requires_grad = False

#    model.module.bart.model.encoder.requires_grad = False
#    model.module.bart.model.decoder.embed_tokens.requires_grad = False
#    model.module.bart.model.decoder.embed_positions.requires_grad = False
#    model.module.bart.model.decoder.layers.requires_grad = False

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader.
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention masks
        #   [2]: labels
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_output_ids = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because
        # accumulating the gradients is "convenient while training RNNs".
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        batch_outputs = model(b_input_ids,
                    attention_mask=b_input_mask,
                    decoder_input_ids=b_input_ids)

        #print("Batch_outputs: ", batch_outputs[0].shape)
        #print("Target: ", b_output_ids.shape)
        #print("Vocab: ", tokenizer.vocab_size)

        target_mask = torch.ones_like(b_output_ids[:, 1:].contiguous()).float()

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.

        #loss = loss_func(batch_outputs[0].view(-1), b_output_ids.view(-1), target_mask, label_smoothing=0.1, reduce="batch")
        loss = loss_func(batch_outputs[0][:, :-1].contiguous(), b_output_ids[:, 1:].contiguous(), target_mask, label_smoothing=0.1, reduce="batch")
        #print(loss)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help preveent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_output_ids = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # batch_outputs = model.module.generate(input_ids=b_input_ids,
            #             #attention_mask=b_input_mask,
            #             num_beams=4,
            #             length_penalty=2.0,
            #             max_length=20,  # +2 from original because we start at step=1 and stop before max_length
            #             min_length=4,  # +1 from original because we start at step=1
            #             no_repeat_ngram_size=3,
            #             repetition_penalty=2,
            #             early_stopping=True,
            #             use_cache = False
            #         )
            #
            batch_logits = model(b_input_ids,
                        attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.

        #print("batch_outputs:", batch_outputs.shape)
        #print("b_output_ids:", b_output_ids.shape)

        #loss = loss_func(batch_logits[0].view(-1, 50264), b_output_ids.view(-1))

        target_mask = torch.ones_like(b_output_ids[:, 1:].contiguous()).float()

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple.

        #loss = loss_func(batch_outputs[0].view(-1), b_output_ids.view(-1), target_mask, label_smoothing=0.1, reduce="batch")
        loss = loss_func(batch_logits[0][:, :-1].contiguous(), b_output_ids[:, 1:].contiguous(), target_mask, label_smoothing=0.1, reduce="batch")
        #print(loss)

        total_val_loss += loss

        # logits = outputs[0]
        #
        # # Move logits and labels to CPU
        # logits = logits.detach().cpu().numpy()
        # label_ids = b_output_ids.to('cpu').numpy()
        #
        # # Calculate the accuracy for this batch of test sentences.
        # tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        #
        # # Accumulate the total accuracy.
        # eval_accuracy += tmp_eval_accuracy
        #
        # Track the number of batches
        nb_eval_steps += 1
        #testing - check what the decoder does with orig inputs
        # in_dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in b_input_ids]
        # dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in batch_outputs]
        # print([(intext,outtext) for intext, outtext in zip(in_dec,dec)])

    # Report the final accuracy for this validation run.
    curr_loss = total_val_loss / len(validation_dataloader)
    print("  Loss: {0:.2f}".format(total_val_loss / len(validation_dataloader)))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

    if curr_loss < prev_val_loss:
        prev_val_loss = curr_loss
        print("")
        #print("Training complete!")
        print("Saving model - epoch, ", epoch_i + 1)
        model.module.save_pretrained(sys.argv[2])
        input_tokenizer.save_pretrained(sys.argv[2])
        output_tokenizer.save_pretrained(sys.argv[2])
