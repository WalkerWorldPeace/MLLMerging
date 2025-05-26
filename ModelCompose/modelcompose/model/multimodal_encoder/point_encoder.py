import torch
import torch.nn as nn
import numpy as np
import os
from transformers.modeling_utils import get_parameter_device, get_parameter_dtype

import yaml
from easydict import EasyDict

from .pointbert.point_encoder import PointTransformer

class PointEncoder(nn.Module):
    def __init__(self, point_encoder, args=None, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.point_encoder_name = point_encoder

        if not delay_load:
            self.load_model()
        else:
            self.load_config()

    def load_config(self):
        point_bert_config_name = "PointTransformer_8192point_2layer" # * default for v1.2, v1.1 uses PointTransformer_base_8192point.yaml
        point_bert_config_addr = os.path.join(os.path.dirname(__file__), "pointbert", f"{point_bert_config_name}.yaml")
        point_bert_config = EasyDict(yaml.safe_load(open(point_bert_config_addr)))
        point_bert_config.model.point_dims = 6
        self.cfg_only = point_bert_config.model

    def load_model(self):
        if self.is_loaded:
            return

        self.load_config()
        checkpoint = torch.load(self.point_encoder_name)
        point_encoder = PointTransformer(self.config, use_max_pool=self.config.use_max_pool)
        point_encoder.load_state_dict(checkpoint)
        point_encoder.eval()
        
        self.point_encoder = point_encoder
        
        self.point_processor = PointCloudProcessor()
        
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, point_clouds):
        return self.point_encoder(point_clouds) # b x 513 x 384

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def config(self):
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.trans_dim if not self.config.use_max_pool else self.config.trans_dim * 2

    @property
    def modal_processor(self):
        return self.point_processor
    
    @property
    def dummy_inputs(self):
        return torch.zeros(1, 8192, 6, device=self.device, dtype=self.dtype)

class PointCloudProcessor:
    
    def __call__(self, pc_files):
        if isinstance(pc_files, str):
            pc_files = [pc_files]

        pc_data = []
        for pc_file in pc_files:
            pc = np.load(pc_file) # N x c
            pc_data.append(pc)
        pc_data = np.stack(pc_data, axis=0)
        
        return torch.from_numpy(pc_data.astype(np.float32))
            
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc
            

if __name__ == '__main__':
    pointencoder = PointEncoder('/yeesuanAI05/thumt/cc/checkpoints/PointLLM/point_bert_v1.2.pt')
    data = pointencoder.modal_processor('/yeesuanAI05/thumt/cc/MITv2/data/multimodal_dataset/PointCloud/8192_npy/05ba73f3f2bb4050988e087f53e98dc9_8192.npy')
    import ipdb; ipdb.set_trace()