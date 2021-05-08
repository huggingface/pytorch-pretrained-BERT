# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team, The Microsoft Research team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pickle
import unittest

from transformers.file_utils import cached_property
from transformers.models.xlm_prophetnet.tokenization_xlm_prophetnet import SPIECE_UNDERLINE, XLMProphetNetTokenizer
from transformers.testing_utils import require_sentencepiece, slow

from .test_tokenization_common import TokenizerTesterMixin


SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/test_sentencepiece.model")


@require_sentencepiece
class XLMProphetNetTokenizationTest(TokenizerTesterMixin, unittest.TestCase):

    tokenizer_class = XLMProphetNetTokenizer
    test_rust_tokenizer = False

    def setUp(self):
        super().setUp()

        # We have a SentencePiece fixture for testing
        tokenizer = XLMProphetNetTokenizer(SAMPLE_VOCAB, keep_accents=True)
        tokenizer.save_pretrained(self.tmpdirname)

    def test_full_tokenizer(self):
        tokenizer = XLMProphetNetTokenizer(SAMPLE_VOCAB, keep_accents=True)

        tokens = tokenizer.tokenize("This is a test")
        self.assertListEqual(tokens, ["▁This", "▁is", "▁a", "▁t", "est"])

        self.assertListEqual(
            tokenizer.convert_tokens_to_ids(tokens),
            [value + tokenizer.fairseq_offset for value in [285, 46, 10, 170, 382]],
        )

        tokens = tokenizer.tokenize("I was born in 92000, and this is falsé.")
        self.assertListEqual(
            tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "9",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "é",
                ".",
            ],
        )
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertListEqual(
            ids,
            [
                value + tokenizer.fairseq_offset
                for value in [8, 21, 84, 55, 24, 19, 7, -9, 602, 347, 347, 347, 3, 12, 66, 46, 72, 80, 6, -9, 4]
            ],
        )

        back_tokens = tokenizer.convert_ids_to_tokens(ids)
        self.assertListEqual(
            back_tokens,
            [
                SPIECE_UNDERLINE + "I",
                SPIECE_UNDERLINE + "was",
                SPIECE_UNDERLINE + "b",
                "or",
                "n",
                SPIECE_UNDERLINE + "in",
                SPIECE_UNDERLINE + "",
                "[UNK]",
                "2",
                "0",
                "0",
                "0",
                ",",
                SPIECE_UNDERLINE + "and",
                SPIECE_UNDERLINE + "this",
                SPIECE_UNDERLINE + "is",
                SPIECE_UNDERLINE + "f",
                "al",
                "s",
                "[UNK]",
                ".",
            ],
        )

    def test_subword_regularization_tokenizer(self) -> None:
        # Subword regularization is only available for the slow tokenizer.
        tokenizer = self.tokenizer_class(
            SAMPLE_VOCAB, keep_accents=True, sp_model_kwargs={"enable_sampling": True, "alpha": 0.1, "nbest_size": -1}
        )

        self.check_subword_sampling(tokenizer)

    def test_pickle_subword_regularization_tokenizer(self) -> None:
        """Google pickle __getstate__ __setstate__ if you are struggling with this."""
        # Subword regularization is only available for the slow tokenizer.
        sp_model_kwargs = {"enable_sampling": True, "alpha": 0.1, "nbest_size": -1}
        tokenizer = self.tokenizer_class(SAMPLE_VOCAB, keep_accents=True, sp_model_kwargs=sp_model_kwargs)
        tokenizer_bin = pickle.dumps(tokenizer)
        del tokenizer
        tokenizer_new = pickle.loads(tokenizer_bin)

        self.assertIsNotNone(tokenizer_new.sp_model_kwargs)
        self.assertTrue(isinstance(tokenizer_new.sp_model_kwargs, dict))
        self.assertEqual(tokenizer_new.sp_model_kwargs, sp_model_kwargs)
        self.check_subword_sampling(tokenizer_new)

    @cached_property
    def big_tokenizer(self):
        return XLMProphetNetTokenizer.from_pretrained("microsoft/xprophetnet-large-wiki100-cased")

    @slow
    def test_tokenization_base_easy_symbols(self):
        symbols = "Hello World!"
        original_tokenizer_encodings = [35389, 6672, 49, 2]
        self.assertListEqual(original_tokenizer_encodings, self.big_tokenizer.encode(symbols))
