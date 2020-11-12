#!/usr/bin/env python3
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

import t5  # noqa: E402
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402
from transformers.convert_t5_v1_1_original_tf_checkpoint_to_pytorch import (  # noqa: E402
    convert_tf_checkpoint_to_pytorch,
)
from transformers.modeling_t5v2 import T5Config, T5v2ForConditionalGeneration  # noqa: E402


path_to_tf_checkpoint = "/home/patrick/hugging_face/mt5/mt5_mesh_tf"


tok = T5Tokenizer.from_pretrained(path_to_tf_checkpoint + "/sentencepiece.model")
tok.save_pretrained(path_to_tf_checkpoint)
config = T5Config.from_pretrained("t5-small")
config.d_ff = 1024
config.num_decoder_layers = 8
config.num_layers = 8
config.num_heads = 6
# comment this line out if only checkpoints for T5v1.1 should be checked
config.vocab_size = 250112

config.save_pretrained(path_to_tf_checkpoint)

convert_tf_checkpoint_to_pytorch(path_to_tf_checkpoint, path_to_tf_checkpoint + "/config.json", path_to_tf_checkpoint)

t5_model = t5.models.MtfModel(
    model_dir=path_to_tf_checkpoint,
    batch_size=1,
    tpu=None,
    sequence_length={"inputs": 64, "targets": 64},
)

vocab_model_path = path_to_tf_checkpoint + "/sentencepiece.model"

# for T5v1.1 one should set `extra_ids=100`.
vocab = SentencePieceVocabulary(vocab_model_path, extra_ids=0)

score = t5_model.score(
    inputs=["Hello there. Let's put more words in more languages than I originally thought."],
    targets=["Hi I am"],
    vocabulary=vocab,
)

model = T5v2ForConditionalGeneration.from_pretrained(path_to_tf_checkpoint, return_dict=True)

input_ids = tok("Hello there", return_tensors="pt").input_ids
labels = tok("Hi I am", return_tensors="pt").input_ids

# input_ids and labels are ok!
loss = model(input_ids, labels=labels).loss
mesh_tf_loss = -(labels.shape[-1] * loss.item())

if mesh_tf_loss - score[0][0] < 1e-4:
    print("Success!")
else:
    print(f"Fail. Mesh TF {mesh_tf_loss} vs. {score[0][0]}")
