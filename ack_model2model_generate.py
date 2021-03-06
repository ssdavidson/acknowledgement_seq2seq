import argparse, csv
from pathlib import Path

import torch
from tqdm import tqdm

from transformers import BertTokenizer, EncoderDecoderModel


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(
    examples: list, out_file: str, model_name: str, batch_size: int = 8, device: str = DEFAULT_DEVICE
):
    fout = Path(out_file).open("w")
#    model = EncoderDecoderModel.from_pretrained(model_name, output_past=True).to(device)
    model = EncoderDecoderModel.from_pretrained(model_name).to(device)
    tokenizer = BertTokenizer.from_pretrained(model_name)

#    max_length = 140
#    min_length = 55

    for batch in tqdm(list(chunks(examples, batch_size))):
        dct = tokenizer.batch_encode_plus(batch, max_length=128, return_tensors="pt", pad_to_max_length=True, add_special_tokens=True)
        summaries = model.generate(
            input_ids=dct["input_ids"].to(device),
            attention_mask=dct["attention_mask"].to(device),
            num_beams=4,
            length_penalty=10.0,
            repetition_penalty = 5.0,
            max_length=20,  # +2 from original because we start at step=1 and stop before max_length
            min_length=3,  # +1 from original because we start at step=1
            no_repeat_ngram_size=3,
            early_stopping=True,
        #    decoder_start_token_id=model.config.decoder.eos_token_id
            decoder_start_token_id=model.config.decoder.pad_token_id
        )
        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        in_ids = dct["input_ids"].to(device)
        in_dec = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in in_ids]
        for input, hypothesis in zip(in_dec, dec):
            fout.write(input + ' ||| ' + hypothesis + "\n")
            fout.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source_path", type=str, help="like cnn_dm/test.source",
    )
    parser.add_argument(
        "output_path", type=str, help="where to save summaries",
    )
    parser.add_argument(
        "model_name", type=str, default="bart-large-cnn", help="like bart-large-cnn",
    )
    parser.add_argument(
        "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
    )
    parser.add_argument(
        "--bs", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()
    tsv_in = csv.reader(open(args.source_path), delimiter = '\t')
    examples = [" " + row[0].rstrip() for row in tsv_in]
    print(examples)
    generate_summaries(examples, args.output_path, args.model_name, batch_size=args.bs, device=args.device)


if __name__ == "__main__":
    run_generate()
