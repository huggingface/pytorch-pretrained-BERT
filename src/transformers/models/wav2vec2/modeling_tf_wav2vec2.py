# coding=utf-8
# Copyright 2021 Spokestack and The HuggingFace Inc. team. All rights reserved.
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
""" TF 2.0 Wav2Vec2 model. """


import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from ...activations_tf import get_tf_activation
from ...file_utils import (
    MULTIPLE_CHOICE_DUMMY_INPUTS,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from ...modeling_tf_outputs import (
    TFBaseModelOutput,
    TFBaseModelOutputWithPooling,
    TFCausalLMOutput,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    get_initializer,
    input_processing,
    keras_serializable,
    shape_list,
)
from ...utils import logging
from .configuration_wav2vec2 import Wav2Vec2Config


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Wav2Vec2Config"
_TOKENIZER_FOR_DOC = "Wav2Vec2Tokenizer"

TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
]


class TFWav2Vec2NoLayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv1D(
            filters=config.conv_dim[layer_id],
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            use_bias=config.conv_bias,
            kernel_initializer=get_initializer(),
        )
        self.activation = tf.keras.layers.Activation(config.feat_extract_activation)

    def call(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2LayerNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.conv = tf.keras.layers.Conv1D(
            filters=config.conv_dim[layer_id],
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            activation=config.feat_extract_activation,
            use_bias=config.conv_bias,
            kernel_initializer=get_initializer(),
        )
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.activation = tf.keras.layers.Activation(config.feature_extract_activation)

    def call(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2GroupNormConvLayer(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, layer_id: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.out_conv_dim = config.conv_dim[layer_id]

        self.conv = tf.keras.layers.Conv1D(
            filters=self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            activation=config.feat_extract_activation,
            use_bias=config.conv_bias,
            kernel_initializer=get_initializer(),
        )
        self.activation = tf.keras.layers.Activation(config.feature_extract_activation)
        self.layer_norm = tfa.layers.GroupNormalization(groups=self.out_conv_dim)

    def call(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2PositionalConvEmbedding(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(
            filters=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            kernel_initializer=get_initializer(),
            padding="same",
        )
        self.activation = tf.keras.layers.Activation(config.feat_extract_activation)

    def call(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class TFWav2Vec2SamePadLayer(tf.keras.layers.Layer):
    pass


class TFWav2Vec2FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if config.feat_extract_norm == "group":
            conv_layers = [TFWav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                TFWav2Vec2NoLayerNormConvLayer(config, layer_id=I + 1)
                for i in range(config.num_feature_extract_layers) - 1
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [
                TFWav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = conv_layers

    def call(self, input_values):
        hidden_states = input_values[:, None]
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states)

        return hidden_states


class TFWav2Vec2FeatureProjection(tf.keras.layers.Layer):
    def __init__(self, config: Wav2Vec2Config, **kwargs):
        super().__init__(**kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.projection = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)

    def call(self, hidden_states: Tensor, input_tensor: Tensor, training: bool = False) -> Tensor:
        hiddent_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        return hidden_states


class TFWav2Vec2Attention(tf.keras.layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias)
        self.v_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias)
        self.q_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias)
        self.out_proj = tf.keras.layers.Dense(embed_dim, use_bias=bias)

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        hidden_states: Tensor,
        key_value_states: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        training: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_length, embed_dim = shape_list(hidden_states)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, val proj
        if is_cross_attention and past_key_value is not None:
            # reuse k, v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self.
        return outputs


class TFWav2Vec2Model:
    pass
