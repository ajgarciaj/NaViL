# --------------------------------------------------------
# NaViL
# Copyright (c) 2025 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from transformers import AutoConfig, LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_navil_vit import NaViLVisionConfig

from navil.model.internlm2.configuration_internlm2 import InternLM2Config
from navil.model.qwen3.configuration_qwen3 import Qwen3VEConfig

logger = logging.get_logger(__name__)


class NaViLChatConfig(PretrainedConfig):
    model_type = 'navil_chat'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            anyres_image_size=True,
            scale_downsample_ratio=0.7071,
            ps_version='v1',
            min_dynamic_patch=256,
            max_dynamic_patch=24576,
            **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing the NaViLVisionConfig with default values.')

        if llm_config is None:
            llm_config = {'architectures': ['InternLM2VEForCausalLM']}
            logger.info('llm_config is None. Initializing the llm_config with default values (`InternLM2VEForCausalLM`).')

        self.vision_config = NaViLVisionConfig(**vision_config)
        self.vision_config.downsample_ratio = downsample_ratio
        if llm_config['architectures'][0] == 'InternLM2VEForCausalLM':
            self.llm_config = InternLM2Config(**llm_config)
        elif llm_config['architectures'][0] == 'Qwen3VEForCausalLM':
            self.llm_config = Qwen3VEConfig(**llm_config)
        else:
            raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template

        self.anyres_image_size = anyres_image_size
        self.scale_downsample_ratio = scale_downsample_ratio
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['pad2square'] = self.pad2square
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        
        output['anyres_image_size'] = self.anyres_image_size
        output['scale_downsample_ratio'] = self.scale_downsample_ratio
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output
