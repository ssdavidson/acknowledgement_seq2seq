import torch
from transformers import BertTokenizer, Model2Model
import load_data

if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#Get Data
inputs, outputs = load_data.get_data(sys.argv[1])

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = Model2Model.from_pretrained('bert-base-uncased')

#encode inputs using BERT tokenizer
input_ids = []
output_ids = []

#input_ids = tokenizer.batch_encode_plus(inputs, max_length=1024, return_tensors='pt')
#output_ids = tokenizer.batch_encode_plus(outputs, max_length=1024, return_tensors='pt')

for in_data, output in zip(inputs, outputs):
     #print(in_data)
     #print(output)
     encoded_input = tokenizer.encode(in_data, add_special_tokens = True, max_length=256,pad_to_max_length=True)
     encoded_output = tokenizer.encode(output, add_special_tokens = True, max_length=256,pad_to_max_length=True)
     input_ids.append(encoded_input)
     output_ids.append(encoded_output)

#add padding to max len
def pad_input(input_ids):
    MAX_LEN = 128
    print('\nPadding/truncating all sentences to %d values...' % MAX_LEN)
    print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
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
