# Code copied from https://github.com/rosewang2008/language_modeling_via_stochastic_processes
# Language modeling via stochastic processes (ICLR Oral 2022)
# https://arxiv.org/abs/2203.11370
"""
@misc{https://doi.org/10.48550/arxiv.2203.11370,
  doi = {10.48550/ARXIV.2203.11370},
  url = {https://arxiv.org/abs/2203.11370},
  author = {Wang, Rose E and Durmus, Esin and Goodman, Noah and Hashimoto, Tatsunori},
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Language modeling via stochastic processes},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

from typing import Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import GPT2PreTrainedModel, GPT2Tokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


class GPT2TimeModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config, model_name):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        if not hasattr(config, "use_contrastive_embeddings"):
            config.use_contrastive_embeddings = False

        if hasattr(config, "cl_latent_dim") and config.cl_latent_dim is not None:
            self.cl2e = nn.Linear(config.cl_latent_dim, self.embed_dim)

        self.cl_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.cl_tokenizer.pad_token = self.cl_tokenizer.eos_token
        self.cl_end_token = self.cl_tokenizer.eos_token_id

        try:
            MAX_NUM_SECTIONS = config.max_num_sections
        except:
            # Error is hit for older models for toy wikisection setup.
            MAX_NUM_SECTIONS = 4
        # NOTE: batch_size 1
        self.section_onehot = torch.FloatTensor(1, MAX_NUM_SECTIONS)

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.section2e = nn.Embedding(MAX_NUM_SECTIONS, self.embed_dim)
        # num sections + 1 null embedding.
        self.sectionNull2e = nn.Embedding(MAX_NUM_SECTIONS + 1, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [GPT2Block(config) for _ in range(config.num_hidden_layers)]
        )
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self._config = config

        # For doing eval generation
        self._transition_cl = False
        self._cur_cl_idx = 0
        self._has_reset = False
        # For doing eval generation
        self._transition_section = False
        self._cur_section_idx = 0

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        self.section2e = self.section2e.to(self.first_device)

        if self._config.use_contrastive_embeddings:
            self.cl2e = self.cl2e.to(self.first_device)
            # self.cl_model = self.cl_model.to(self.first_device)
            self.cl_tokenizer = self.cl_tokenizer.to(self.first_device)

        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        self.section2e = self.section2e.to("cpu")
        if self._config.use_contrastive_embeddings:
            self.cl2e = self.cl2e.to("cpu")
            # self.cl_model = self.cl_model.to("cpu")
            self.cl_tokenizer = self.cl_tokenizer.to("cpu")

        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def cl_tokenize_text(self, text):
        output = self.cl_tokenizer(
            text,
            padding=True,
            return_tensors="pt",
        )
        input_ids = output["input_ids"]  # .squeeze(0)
        attention_mask = output["attention_mask"]  # .squeeze(0)
        eos_input_ids = torch.tensor([[self.cl_end_token] * input_ids.shape[0]])
        eos_attention = torch.tensor([[0] * input_ids.shape[0]])
        input_ids = torch.cat((input_ids, eos_input_ids.T), dim=1)
        attention_mask = torch.cat((attention_mask, eos_attention.T), dim=1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return input_ids.to(device), attention_mask.to(device)

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def _get_cl_embeddings(self, raw_text, cl_feats, seq_cl_feats, input_ids, seq_len):
        # NOTE assuming batch size 1
        generated_by_raw_text = False

        if seq_cl_feats is not None and cl_feats is not None:  # Used in evaluation
            if input_ids.shape[0] > 1 and (
                not isinstance(self._transition_cl, list)
                or not isinstance(self._cur_cl_idx, list)
            ):  # beam
                self._transition_cl = [False] * input_ids.shape[0]
                self._cur_cl_idx = [0] * input_ids.shape[0]
                cl_feats = cl_feats.expand(input_ids.shape[0], cl_feats.shape[-1])
                self._last_beam_cl_feats = cl_feats

            else:  # non beam
                if (
                    "wikisection" in self.config.dataset_name
                    or "wikihow" in self.config.dataset_name
                    or "stories" in self.config.dataset_name
                ):
                    if input_ids[0][0] == self.special_tokens[-1]:  # " . " token
                        try:
                            num_feats = seq_cl_feats.shape[0] - 1
                        except:
                            num_feats = len(seq_cl_feats) - 1
                        self._cur_cl_idx = min(self._cur_cl_idx + 1, num_feats)
                    cl_feats = seq_cl_feats[self._cur_cl_idx]
                elif "taskmaster" in self.config.dataset_name:
                    if input_ids[0][0] in self.special_tokens:
                        if not self._has_reset:  # don't iterate past
                            self._has_reset = True
                        else:
                            self._cur_cl_idx = min(
                                self._cur_cl_idx + 1, seq_cl_feats.shape[0] - 1
                            )
                    cl_feats = seq_cl_feats[self._cur_cl_idx]
                else:
                    self._cur_cl_idx = min(
                        self._cur_cl_idx + 1, seq_cl_feats.shape[0] - 1
                    )
                    cl_feats = seq_cl_feats[self._cur_cl_idx]

        if cl_feats is None and seq_cl_feats is not None:
            cl_feats = seq_cl_feats

        cl_embeds = self.cl2e(cl_feats)

        if generated_by_raw_text:
            cl_embeds = cl_embeds.unsqueeze(0)
        else:
            if input_ids.shape[0] == 1:
                cl_embeds = cl_embeds.expand(1, seq_len, 768)
            else:  # beam
                cl_embeds = cl_embeds.unsqueeze(1)

        return cl_embeds

    def _get_section_ids(self, input_ids, section_ids, seq_section_ids):
        # desired shape: section ids = [batch_size, 1]
        seq_len = input_ids.shape[1]
        if seq_section_ids is not None:  # Used in evaluation
            if input_ids.shape[0] > 1 and (
                not isinstance(self._transition_section, list)
                or not isinstance(self._cur_section_idx, list)
            ):  # beam
                self._transition_section = [False] * input_ids.shape[0]
                self._cur_section_idx = [0] * input_ids.shape[0]
                section_ids = section_ids.expand(
                    input_ids.shape[0], section_ids.shape[-1]
                )
                self._last_section_ids = section_ids

            if input_ids.shape[0] > 1:
                # TODO off by one - need to start replacing on the second 764 mention
                section_ids = torch.clone(self._last_section_ids)
                for seq_idx, beam_seq in enumerate(input_ids):
                    if beam_seq[-1] == 764:  # eos
                        self._transition_section[seq_idx] = True
                    elif self._transition_section[seq_idx]:  # last id was eos
                        if (
                            self._cur_section_idx[seq_idx] + 1
                            < seq_section_ids.shape[0]
                        ):
                            self._cur_section_idx[seq_idx] += 1
                            section_ids[seq_idx] = seq_section_ids[
                                self._cur_section_idx[seq_idx]
                            ]
                self._last_beam_section_ids = torch.clone(section_ids)

            else:  # non beam
                section_ids = seq_section_ids[self._cur_section_idx]
                if len(seq_section_ids) != self._cur_section_idx + 1:
                    self._cur_section_idx += 1
                section_ids = section_ids.expand(1, seq_len)

        elif section_ids is None:
            section_ids = torch.zeros((1, seq_len))
            end_idx = seq_len
            _break = False
            for section_num, section_token in enumerate(self.special_tokens[::-1]):
                if section_token == self.special_tokens[0]:  # first section
                    start_idx = 0
                else:
                    start_idx = (input_ids == section_token).nonzero(as_tuple=True)[-1]
                    if not start_idx.shape[0]:  # empty, could not be found.
                        start_idx = 0
                        _break = True
                section_ids[:, start_idx:end_idx] = 3 - section_num
                if _break:
                    break
                end_idx = start_idx
            section_ids = section_ids.to(self.device).long()
        else:  # have section_ids (1, 1)
            section_ids = section_ids.expand(-1, seq_len)  # (1, seq_len)

        return section_ids

    def forward(
        self,
        input_ids=None,
        raw_text=None,
        cl_feats=None,
        seq_cl_feats=None,
        seq_section_ids=None,
        section_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # fulldoc & wikisection
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        hidden_states = inputs_embeds + position_embeds

        # Use embeddings from CL training.
        if self._config.use_contrastive_embeddings:
            if self._config.use_section_ids:
                raise ValueError(
                    "contrastive embeddings should not be used at same time as section ids"
                )
            cl_embeds = self._get_cl_embeddings(
                raw_text=raw_text,
                cl_feats=cl_feats,
                seq_cl_feats=seq_cl_feats,
                input_ids=input_ids,
                seq_len=inputs_embeds.shape[1],
            )
            hidden_states = hidden_states + cl_embeds

        # Do section embeddings
        if self._config.use_section_ids:
            if self._config.use_contrastive_embeddings:
                raise ValueError(
                    "contrastive embeddings should not be used at same time as section ids"
                )
            # section ids = [batch_size, 1]
            section_ids = self._get_section_ids(
                input_ids=input_ids,
                section_ids=section_ids,
                seq_section_ids=seq_section_ids,
            )
            if (
                hasattr(self._config, "use_section_null")
                and self._config.use_section_null
            ):
                section_embeds = self.sectionNull2e(section_ids)
            else:
                section_embeds = self.section2e(section_ids)
            hidden_states = hidden_states + section_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(
                        past_state.to(hidden_states.device) for past_state in layer_past
                    )
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (
                        outputs[3 if use_cache else 2],
                    )

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2TimeLMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"attn.masked_bias",
        r"attn.bias",
        r"lm_head.weight",
    ]

    def __init__(self, config, model_name="gpt2"):
        super().__init__(config)
        self.transformer = GPT2TimeModel(config, model_name)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        result = {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        if "section_ids" in kwargs:
            result["section_ids"] = kwargs["section_ids"]
        if "raw_text" in kwargs:
            result["raw_text"] = kwargs["raw_text"]
        if "cl_feats" in kwargs:
            result["cl_feats"] = kwargs["cl_feats"]
        if "seq_cl_feats" in kwargs:
            result["seq_cl_feats"] = kwargs["seq_cl_feats"]
        if "seq_section_ids" in kwargs:
            result["seq_section_ids"] = kwargs["seq_section_ids"]

        return result

    def forward(
        self,
        input_ids=None,
        raw_text=None,
        seq_cl_feats=None,
        seq_section_ids=None,
        cl_feats=None,
        section_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.transformer(
            input_ids,
            raw_text=raw_text,
            cl_feats=cl_feats,
            seq_cl_feats=seq_cl_feats,
            seq_section_ids=seq_section_ids,
            section_ids=section_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )
