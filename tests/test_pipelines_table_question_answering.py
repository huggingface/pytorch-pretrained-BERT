# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import unittest

from transformers.pipelines import Pipeline, pipeline
from transformers.testing_utils import require_pandas, require_torch, require_torch_scatter

from .test_pipelines_common import CustomInputPipelineCommonMixin


@require_torch_scatter
@require_torch
@require_pandas
class TQAPipelineTests(CustomInputPipelineCommonMixin, unittest.TestCase):
    pipeline_task = "table-question-answering"
    pipeline_running_kwargs = {
        "padding": "max_length",
    }
    small_models = [
        "lysandre/tiny-tapas-random-wtq",
        "lysandre/tiny-tapas-random-sqa",
    ]
    large_models = ["nielsr/tapas-base-finetuned-wtq"]  # Models tested with the @slow decorator
    valid_inputs = [
        {
            "table": {
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            "query": "how many movies has george clooney played in?",
        },
        {
            "table": {
                "actors": ["brad pitt", "leonardo di caprio", "george clooney"],
                "age": ["56", "45", "59"],
                "number of movies": ["87", "53", "69"],
                "date of birth": ["7 february 1967", "10 june 1996", "28 november 1967"],
            },
            "query": ["how many movies has george clooney played in?", "how old is he?", "what's his date of birth?"],
        },
        {
            "table": {
                "Repository": ["Transformers", "Datasets", "Tokenizers"],
                "Stars": ["36542", "4512", "3934"],
                "Contributors": ["651", "77", "34"],
                "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
            },
            "query": [
                "What repository has the largest number of stars?",
                "Given that the numbers of stars defines if a repository is active, what repository is the most active?",
                "What is the number of repositories?",
                "What is the average number of stars?",
                "What is the total amount of stars?",
            ],
        },
    ]

    def _test_pipeline(self, nlp: Pipeline):
        output_keys = {"answer", "coordinates", "cells"}
        valid_inputs = self.valid_inputs
        invalid_inputs = [
            {"query": "What does it do with empty context ?", "table": ""},
            {"query": "What does it do with empty context ?", "table": None},
        ]
        self.assertIsNotNone(nlp)

        mono_result = nlp(valid_inputs[0])
        self.assertIsInstance(mono_result, dict)

        for key in output_keys:
            self.assertIn(key, mono_result)

        multi_result = nlp(valid_inputs)
        self.assertIsInstance(multi_result, list)
        for result in multi_result:
            self.assertIsInstance(result, (list, dict))

        for result in multi_result:
            if isinstance(result, list):
                for _result in result:
                    for key in output_keys:
                        self.assertIn(key, _result)
            else:
                for key in output_keys:
                    self.assertIn(key, result)
        for bad_input in invalid_inputs:
            self.assertRaises(ValueError, nlp, bad_input)
        self.assertRaises(ValueError, nlp, invalid_inputs)

    def test_aggregation(self):
        nlp = pipeline(
            "table-question-answering",
            model="lysandre/tiny-tapas-random-wtq",
            tokenizer="lysandre/tiny-tapas-random-wtq",
        )
        self.assertIsInstance(nlp.model.config.aggregation_labels, dict)
        self.assertIsInstance(nlp.model.config.no_aggregation_label_index, int)

        mono_result = nlp(self.valid_inputs[0])
        multi_result = nlp(self.valid_inputs)

        self.assertIn("aggregator", mono_result)

        for result in multi_result:
            if isinstance(result, list):
                for _result in result:
                    self.assertIn("aggregator", _result)
            else:
                self.assertIn("aggregator", result)

    def test_aggregation_with_sequential(self):
        nlp = pipeline(
            "table-question-answering",
            model="lysandre/tiny-tapas-random-wtq",
            tokenizer="lysandre/tiny-tapas-random-wtq",
        )
        self.assertIsInstance(nlp.model.config.aggregation_labels, dict)
        self.assertIsInstance(nlp.model.config.no_aggregation_label_index, int)

        mono_result = nlp(self.valid_inputs[0], sequential=True)
        multi_result = nlp(self.valid_inputs, sequential=True)

        self.assertIn("aggregator", mono_result)

        for result in multi_result:
            if isinstance(result, list):
                for _result in result:
                    self.assertIn("aggregator", _result)
            else:
                self.assertIn("aggregator", result)

    def test_sequential(self):
        nlp = pipeline(
            "table-question-answering",
            model="lysandre/tiny-tapas-random-sqa",
            tokenizer="lysandre/tiny-tapas-random-sqa",
        )
        sequential_mono_result_0 = nlp(self.valid_inputs[0], sequential=True)
        sequential_mono_result_1 = nlp(self.valid_inputs[1], sequential=True)
        sequential_multi_result = nlp(self.valid_inputs, sequential=True)
        mono_result_0 = nlp(self.valid_inputs[0])
        mono_result_1 = nlp(self.valid_inputs[1])
        multi_result = nlp(self.valid_inputs)

        # First valid input has a single question, the dict should be equal
        self.assertDictEqual(sequential_mono_result_0, mono_result_0)

        # Second valid input has several questions, the questions following the first one should not be equal
        self.assertNotEqual(sequential_mono_result_1, mono_result_1)

        # Assert that we get the same results when passing in several sequences.
        for index, (sequential_multi, multi) in enumerate(zip(sequential_multi_result, multi_result)):
            if index == 0:
                self.assertDictEqual(sequential_multi, multi)
            else:
                self.assertNotEqual(sequential_multi, multi)
