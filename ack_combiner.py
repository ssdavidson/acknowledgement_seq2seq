from transformers import PreTrainedEncoderDecoder, GPT2Model, GPT2Config, GPT2Tokenizer
import numpy as np

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

#config = GPT2Config.from_pretrained('gpt2-large')
#gpt2_decoder = GPT2Model(config)

model = PreTrainedEncoderDecoder.from_pretrained('bert-base-uncased', 'gpt-large')

encoder_input_ids = tokenizer.encode(input_text)
