import torch
from python_src.models.model_utils import TokenClassifierCustomOutput
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5EncoderModel, T5Config, T5Model
from transformers.modeling_outputs import BaseModelOutput
import warnings


class T5TokenClassificationConfig(T5Config):
    def __init__(self, num_labels=2, hidden_dropout_prob=0.1, classifier_dropout=None, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout = classifier_dropout


class T5EncoderTokenClassification(T5EncoderModel):

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.token_classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = getattr(config, "classifier_dropout", None)
        elif getattr(config, "hidden_dropout_prob", None) is not None:
            classifier_dropout = getattr(config, "hidden_dropout_prob", None)
        else:
            classifier_dropout = 0.0

        self.dropout = nn.Dropout(classifier_dropout)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            verbs_sequence=None, nouns_sequence=None, verb_noun_id_labels=None, nsp_labels=None
    ):

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = encoder_outputs[0]
        sequence_output = self.dropout(last_hidden_states)
        logits = self.token_classifier(sequence_output)

        loss_fct = CrossEntropyLoss(reduction="mean")
        loss = calculate_loss(self.num_labels, attention_mask, labels, logits, loss_fct)

        if not return_dict:
            output = (logits,) + encoder_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierCustomOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
HEAD_MASK_WARNING_MSG = "The input argument `head_mask` was split into two arguments `head_mask` and " \
                        "`decoder_head_mask`. Currently, `decoder_head_mask` is set to copy `head_mask`, " \
                        "but this feature is deprecated and will be removed in future versions. If you do not " \
                        "want to use any `decoder_head_mask` now, please set " \
                        "`decoder_head_mask = torch.ones(num_layers, num_heads)`."


class T5EncoderDecoderTokenClassification(T5Model):

    def __init__(self, config: T5Config, **kwargs):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.token_classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = getattr(config, "classifier_dropout", None)
        elif getattr(config, "hidden_dropout_prob", None) is not None:
            classifier_dropout = getattr(config, "hidden_dropout_prob", None)
        else:
            classifier_dropout = 0.0

        self.dropout = nn.Dropout(classifier_dropout)

        # Initialize weights and apply final processing
        # self.post_init()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
            verbs_sequence=None, nouns_sequence=None, verb_noun_id_labels=None, nsp_labels=None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
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

        decoder_input_ids = input_ids.clone()

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_states = decoder_outputs.last_hidden_state
        sequence_output = self.dropout(last_hidden_states)
        logits = self.token_classifier(sequence_output)

        loss_fct = CrossEntropyLoss(reduction="mean")
        loss = calculate_loss(self.num_labels, attention_mask, labels, logits, loss_fct)

        if not return_dict:
            output = (logits, decoder_outputs.last_hidden_state, decoder_outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierCustomOutput(
            loss=loss,
            logits=logits,
            hidden_states=decoder_outputs.last_hidden_state,
            attentions=decoder_outputs.attentions,
        )


def calculate_loss(num_labels, attention_mask, labels, logits, loss_fct):
    loss = None
    if labels is not None:
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

    return loss
