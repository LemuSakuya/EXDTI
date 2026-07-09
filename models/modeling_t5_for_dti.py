import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
import warnings

from transformers.models.t5.modeling_t5 import (
    T5PreTrainedModel,
    T5Config,
    T5Stack,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutput
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.utils.logging import get_logger

# These constants were removed in transformers v5.x — provide empty placeholders
T5_START_DOCSTRING = ""
T5_INPUTS_DOCSTRING = ""
PARALLELIZE_DOCSTRING = ""
DEPARALLELIZE_DOCSTRING = ""
_CONFIG_FOR_DOC = "T5Config"

# model_parallel_utils was removed in transformers v5.x — provide minimal stubs
def assert_device_map(device_map, num_layers):
    pass

def get_device_map(num_layers, devices):
    return {}

logger = get_logger("transformers")


class InteractionModule(nn.Module):
    def __init__(self, hid_dim):
        super(InteractionModule, self).__init__()
        self.linear = nn.Linear(hid_dim, hid_dim)

    def forward(self, inputs, mask1=None, mask2=None):
        transformed_inputs = self.linear(inputs)
        interaction_matrix = torch.einsum("bik,bjk->bij", inputs, transformed_inputs)

        if mask1 is not None and mask2 is not None:
            mask1 = mask1.unsqueeze(2)
            mask2 = mask2.unsqueeze(1)
            mask = mask1 * mask2
            interaction_matrix *= mask

        output = interaction_matrix.mean(dim=(1, 2))
        return output


@add_start_docstrings(
    """T5 Model with non-linear mapping layer on top for DTI classification.""",
    T5_START_DOCSTRING
)
class T5ForDTIClassification(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, self.shared)

        classifier_dropout = (
            config.classifier_dropout if hasattr(config, "classifier_dropout") else config.dropout_rate
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = InteractionModule(config.d_model)

        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn("Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.", FutureWarning)
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.classifier = self.classifier.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        drug_mask: Optional[torch.FloatTensor] = None,
        target_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], SequenceClassifierOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        logits = self.classifier(hidden_states, drug_mask, target_mask)

        loss = None
        if labels is not None:
            pred_criterion = nn.BCEWithLogitsLoss()
            labels = labels.float()
            loss = pred_criterion(logits, labels.squeeze())

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions
        )
