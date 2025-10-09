# --------------------------------------------------------
# NaViL
# Copyright (c) 2025 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import warnings
from typing import Any, List, Optional, Tuple, Union
import copy

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from .configuration_navil_chat import NaViLChatConfig
from .modeling_navil_vit_anyres import NaViLVisionModelAnyRes

from navil.conversation import get_conv_template
from navil.model.internlm2.modeling_internlm2_ve import InternLM2VEForCausalLM
from navil.model.qwen3.modeling_qwen3_ve import Qwen3VEForCausalLM
from navil.model.internlm2.modeling_internlm2_ve import InternLM2RMSNorm
from navil.model.qwen2_vl import Qwen2VLImageProcessor
from navil.train.constants import (
    SPECIAL_TOKEN_LIST,
    IMG_CONTEXT_TOKEN, IMG_END_TOKEN, IMG_START_TOKEN, IMG_UNCOND_TOKEN,
    VAE_MEAN, VAE_STD,
)

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))



@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

    log_dict: Optional[dict] = None


class NaViL(PreTrainedModel):
    config_class = NaViLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['NaViLVisionModelAnyRes', 'InternLM2DecoderLayer', 'Qwen3DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: NaViLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)
        self.config = config

        assert version_cmp(transformers.__version__, '4.51.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.patch_aspect_ratio = 1.0
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'init - image_size: {image_size}, patch_size: {patch_size}, num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = NaViLVisionModelAnyRes(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            llm_config = config.llm_config
            if config.llm_config.architectures[0] == 'InternLM2VEForCausalLM':
                self.language_model = InternLM2VEForCausalLM(llm_config)
            elif config.llm_config.architectures[0] == 'Qwen3VEForCausalLM':
                self.language_model = Qwen3VEForCausalLM(llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.img_start_token_id = None
        self.img_end_token_id = None
        self.img_uncond_token_id = None
        self.img_line_break_token_id = None
        self.img_frame_break_token_id = None
        self.pad_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        
        min_pixels = config.min_dynamic_patch * (patch_size ** 2)
        max_pixels = config.max_dynamic_patch * (patch_size ** 2)
        down_sample_ratio = config.vision_config.downsample_ratio
        self.image_processor = Qwen2VLImageProcessor(
                do_resize=False,
                do_pad=True,
                do_rescale=True,
                do_normalize=True,
                image_mean=VAE_MEAN,
                image_std=VAE_STD,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                patch_size=patch_size,
                temporal_patch_size=1,
                merge_size=int(1.0 / down_sample_ratio),
            )

        ##### ---- Special token embeddings ---- #####
        self.special_token_embedding = nn.Embedding(len(SPECIAL_TOKEN_LIST), config.llm_config.hidden_size)
        self.special_token_list = copy.deepcopy(SPECIAL_TOKEN_LIST)
        self.special_token_id_list = None # Remember to initialize this in the training script after tokenizer is loaded

        self.group = None # Distributed group. Remember to set this in the training script

    def init_special_token_ids(self, tokenizer):
        special_token_id_list = []
        for token in SPECIAL_TOKEN_LIST:
            special_token_id_list.append(tokenizer.convert_tokens_to_ids(token))
        self.special_token_id_list = special_token_id_list

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.img_uncond_token_id = tokenizer.convert_tokens_to_ids(IMG_UNCOND_TOKEN)

    def replace_img_special_tokens(self, input_embeds, input_ids):
        assert self.special_token_id_list is not None, "model's special_token_id_list is not initialized"
        for i, token_id in enumerate(self.special_token_id_list):
            token_pos = input_ids == token_id
            input_embeds[token_pos] = input_embeds[token_pos] * 0.0 + self.special_token_embedding.weight[i]

        return input_embeds

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, Qwen2RMSNorm, InternLM2RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            generation_modality: Optional[int] = 0,
            statistics: Optional[torch.LongTensor] = None,
            loss_weight: Optional[List] = None,
            loss_reduction_all_gather: Optional[bool] = False,
            padding_type: Optional[str] = None,
            type_ids: Optional[torch.LongTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            # cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        ignore_flag = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)

        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()
        input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)

        if video_grid_thw is not None:
            grid_thw = video_grid_thw
        else:
            grid_thw = image_grid_thw
        vit_embeds, vit_embeds_ori = self.extract_feature(pixel_values, grid_thw)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_embeds_ori = vit_embeds_ori[image_flags == 1]
        vit_batch_size = image_flags.sum().item()

        log_dict_keys = [
            "text_loss", "text_acc1",
        ]
        log_dict = {k: torch.tensor(0.0, device=self.device) for k in log_dict_keys}
        return_feature_scale = True

        B, N, C = input_embeds.shape
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            # ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}', force=True)
            n_token = selected.sum()
            if n_token > vit_embeds.shape[0]:
                selected = selected.view(-1, selected.shape[-1])  # 确保是 [B, N] 形状
                batch_size = selected.shape[0]
                max_visual_tokens = vit_embeds.shape[0] // batch_size  # 每个批次可用的视觉特征数量
                for i in range(batch_size):
                    # 获取当前批次中的图像标记位置
                    curr_selected = selected[i]
                    # 只保留前 max_visual_tokens 个标记位置
                    curr_indices = torch.where(curr_selected)[0][:max_visual_tokens]
                    # 更新选择标记
                    selected[i] = False
                    selected[i, curr_indices] = True
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        # input_embeds = input_embeds.reshape(B, N, C)
        visual_token_mask = (selected + (input_ids == self.img_start_token_id))

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            visual_token_mask=visual_token_mask,
            generation_modality=generation_modality,
            padding_type=padding_type, # or self.train_padding_type,
            skip_lm_head=False, # imgen
            return_feature_scale=return_feature_scale,
        )
        logits = outputs.logits   # B, N, C

        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(loss_weight, dtype=torch.float32, device=labels.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG, group=self.group)

            pred_ids = shift_logits.argmax(dim=-1)
            pred_acc = 100.0 * ((shift_labels == pred_ids) * (shift_labels != -100)).sum() / (shift_labels != -100).sum()

            log_dict.update({
                "text_loss": ((loss * shift_weights).sum() / shift_weights_sum).detach(),
                "text_acc1": pred_acc
            })
            
            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum

            if ignore_flag:
                loss = loss * 0.0

        elif labels is not None:
            # To reduce gpu memory, remove the image parts of the logits and labels
            shift_selected = (input_ids == self.img_context_token_id)[..., :-1]
            shift_logits = logits[..., :-1, :][~shift_selected]
            shift_labels = labels[..., 1:][~shift_selected]

            # Shift so that tokens < n predict n
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            pred_ids = shift_logits.argmax(dim=-1)
            pred_acc = 100.0 * ((shift_labels == pred_ids) * (shift_labels != -100)).sum() / (shift_labels != -100).sum()

            log_dict.update({
                "text_loss": loss.mean().detach(),
                "text_acc1": pred_acc
            })

            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if return_feature_scale:
            log_dict["feature_scale"] = {
                "image": outputs.feature_scale[0],
                "text": outputs.feature_scale[1],
            }

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            log_dict=log_dict
        )

    def extract_feature(self, pixel_values, grid_thw=None):
 
        if grid_thw is not None:
            grid_thw = grid_thw.to(pixel_values.device)

        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
            grid_thw=grid_thw
            ).last_hidden_state

        vit_embeds = pixel_shuffle_v2(vit_embeds, scale_factor=self.downsample_ratio, patch_aspect_ratio=self.patch_aspect_ratio)

        vit_embeds_after_mlp = self.mlp1(vit_embeds)

        return vit_embeds_after_mlp, vit_embeds

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, num_scales: list = [2],
             IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             IMG_LINE_BREAK_TOKEN='<IMG_LINE_BREAK>', IMG_FRAME_BREAK_TOKEN='<IMG_FRAME_BREAK>',
             anyres_image_size=True,
             verbose=False,
        ):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' * len(num_scales) + question

        if num_patches_list is None:
            assert not anyres_image_size, "Please provide `num_patches_list` when anyres_image_size is True."
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or anyres_image_size or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id
        img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_start_token_id = img_start_token_id
        self.img_line_break_token_id = tokenizer.convert_tokens_to_ids(IMG_LINE_BREAK_TOKEN)
        self.img_frame_break_token_id = tokenizer.convert_tokens_to_ids(IMG_FRAME_BREAK_TOKEN)

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        if anyres_image_size:
            merge_size = int(1.0 / self.downsample_ratio)
            for image_idx in range(len(num_scales)):
                num_scales_prev = sum(num_scales[:image_idx])
                num_scale = num_scales[image_idx]
                _num_image_token_list = num_patches_list[num_scales_prev:num_scales_prev + num_scale]
                image_tokens = f"{IMG_START_TOKEN}"
                for i in range(len(_num_image_token_list)):
                    _image_tokens = ""
                    t, h, w = _num_image_token_list[i][0], _num_image_token_list[i][1] // merge_size, _num_image_token_list[i][2] // merge_size
                    for _ in range(t):
                        for _ in range(h):
                            _image_tokens += f"{IMG_CONTEXT_TOKEN * w}{IMG_LINE_BREAK_TOKEN}"
                        _image_tokens += f"{IMG_FRAME_BREAK_TOKEN}"
                    image_tokens += _image_tokens
                image_tokens += f"{IMG_END_TOKEN}"
                query = query.replace('<image>', image_tokens, 1)
        else:
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=num_patches_list,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        # fix for InternLM2-base (textvqa)
        response = response.replace("<|im_end|", "")
        response = response.replace("<|im_end", "")
        response = response.replace("<|im", "")
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(IMG_LINE_BREAK_TOKEN, '')
            query_to_print = query_to_print.replace(IMG_FRAME_BREAK_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)

            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None

        grid_thw = image_grid_thw

        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, vit_embeds_ori = self.extract_feature(pixel_values, grid_thw)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)
            B, N, C = input_embeds.shape
            # input_embeds = input_embeds.reshape(B * N, C)

            # input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id) # B, N
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            # input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            input_embeds = self.replace_img_special_tokens(input_embeds, input_ids)
            selected = None
        
        # input_embeds = self.replace_special_tokens(input_embeds, input_ids)
        visual_token_mask = selected + (input_ids == self.img_start_token_id) if selected is not None else None

        position_ids = None
        generate_kwargs['position_ids'] = position_ids
        
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            visual_token_mask=visual_token_mask,
            **generate_kwargs,
        )

        return outputs


def pixel_shuffle_v2(x, scale_factor=0.5, patch_aspect_ratio=1.0):
    # input shape: N, L, C or N, H, W, C
    # output shape: N, L * (scale_factor ** 2), C / (scale_factor ** 2)
    
    if x.ndim == 3:
        n, l, c = x.size()
        h = w = int(l ** 0.5)
        # N, L, C --> N, H, W, C
        x = x.reshape(n, h, w, c)

    n, h, w, c = x.size()

    h_scale_factor = scale_factor * (patch_aspect_ratio ** 0.5)
    w_scale_factor = scale_factor / (patch_aspect_ratio ** 0.5)

    # N, H, W, C --> N, H, W * w_scale_factor, C // w_scale_factor
    x = x.reshape(n, h, int(w * w_scale_factor), int(c / w_scale_factor))
    # N, H, W * w_scale_factor, C // w_scale_factor --> N, W * w_scale_factor, H, C // w_scale_factor
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, W * w_scale_factor, H, C // w_scale_factor --> N, W * w_scale_factor, H * h_scale_factor, C // (w_scale_factor * h_scale_factor)
    x = x.reshape(n, int(w * w_scale_factor), int(h * h_scale_factor), int(c / (w_scale_factor * h_scale_factor)))
    # N, W * w_scale_factor, H * h_scale_factor, C // (w_scale_factor * h_scale_factor) --> N, H * h_scale_factor, W * w_scale_factor, C // (w_scale_factor * h_scale_factor)
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * h_scale_factor, W * w_scale_factor, C // (w_scale_factor * h_scale_factor) --> N, L * (scale_factor ** 2), C // (scale_factor ** 2)
    x = x.reshape(n, int(h * h_scale_factor * w * w_scale_factor), int(c / (h_scale_factor * w_scale_factor)))

    return x
