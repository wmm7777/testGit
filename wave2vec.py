import torch
from torch import nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import time

class WAVE2VEC(nn.Module):
    def __init__(self, feature_dim, num_classes=4, size='base'):
        super(WAVE2VEC, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(r"/data/wumeimei/inspection_train/fackbook/wav2vec2-base")
        # self.text_feature_affine = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, feature_dim)
        # )

    def forward(self, audio, get_cls=False):
        # logits, hidden_states = self.albert(**text, output_hidden_states=True)
        output = self.wav2vec(audio)

        return output.last_hidden_state