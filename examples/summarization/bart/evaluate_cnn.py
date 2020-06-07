import argparse
from pathlib import Path

import torch
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_summaries(
    examples: list, out_file: str, model_name: str, batch_size: int = 8, device: str = DEFAULT_DEVICE
):
    fout = Path(out_file).open("w", encoding="utf-8")
    if "t5" in model_name:
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        # update config with summarization specific params
        task_specific_params = model.config.task_specific_params
        if task_specific_params is not None:
            model.config.update(task_specific_params.get("summarization", {}))
    else:
        model = BartForConditionalGeneration.from_pretrained(model_name).to(device)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    max_length = 140
    min_length = 55

    for batch in tqdm(list(chunks(examples, batch_size))):
        if "t5" in model_name:
            batch = [model.config.prefix + text for text in batch]
        dct = tokenizer.batch_encode_plus(batch, max_length=1024, return_tensors="pt", pad_to_max_length=True)

        input_ids = dct["input_ids"].to(device)
        attention_mask = dct["attention_mask"].to(device)
        if "t5" not in model_name:
            summaries = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                length_penalty=2.0,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=min_length + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                early_stopping=True,
                decoder_start_token_id=model.config.eos_token_id,
            )
        else:
            summaries = model.generate(input_ids=input_ids, attention_mask=attention_mask)

        dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
        for hypothesis in dec:
            fout.write(hypothesis + "\n")
            fout.flush()


def calculate_rouge(output_lns, reference_lns, score_path):
    score_file = Path(score_path).open("w")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    score_file.write(
        "ROUGE_1: \n{} \n\n ROUGE_2: \n{} \n\n ROUGE_L: \n{} \n\n".format(
            result["rouge1"], result["rouge2"], result["rougeL"]
        )
    )


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path", type=str, help="like cnn_dm/test.source or cnn_dm/test_articles_input.txt",
    )
    parser.add_argument(
        "output_path", type=str, help="where to save summaries",
    )
    parser.add_argument(
        "model_name",
        type=str,
        default="facebook/bart-large-cnn",
        help="like bart-large-cnn,'t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b",
    )
    parser.add_argument("--reference_path", type=str, required=False, help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument(
        "--score_path", type=str, required=False, help="where to save the rouge score",
    )
    parser.add_argument(
        "--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.",
    )
    parser.add_argument(
        "--bs", type=int, default=8, required=False, help="batch size: how many to summarize at a time",
    )
    args = parser.parse_args()
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip() for x in open(args.input_path).readlines()]

    generate_summaries(examples, args.output_path, args.model_name, batch_size=args.bs, device=args.device)
    if args.score_path is not None:
        output_lns = [x.rstrip() for x in open(args.output_path).readlines()]
        reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()]

        calculate_rouge(output_lns, reference_lns, args.score_path)


if __name__ == "__main__":
    run_generate()
