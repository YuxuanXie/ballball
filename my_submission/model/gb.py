# Model taken from https://arxiv.org/pdf/1810.08647.pdf,
# INTRINSIC SOCIAL MOTIVATION VIA CAUSAL
# INFLUENCE IN MULTI-AGENT RL


# model is a single convolutional layer with a kernel of size 3, stride of size 1, and 6 output
# channels. This is connected to two fully connected layers of size 32 each

import imp
import math
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
import torch.nn.functional as F


class ScaleDotProductionAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, mask=None):
        d_model = q.shape[-1]
        attn = (torch.matmul(q, k.permute(0,2,1)))/math.sqrt(d_model)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        entity_embedding_prob = F.softmax(attn, dim=-1)
        
        entity_embedding = torch.matmul(entity_embedding_prob, v)
        return entity_embedding

class TorchRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name,
                fc_size=64,
                lstm_state_size=64):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                        name)


        # Holds the current "base" output (before logits layer).
        self._features = None

        custom_model_config = model_config["custom_model_config"]

        self.obs_size = custom_model_config["obs_shape"]
        self.action_type_shape = 4
        self.entity_shape = custom_model_config["entity_shape"]
        self.obs_embedding_size, self.entity_embedding_size = custom_model_config["obs_embedding_size"], custom_model_config["entity_embedding_size"]
        self.all_embedding_size = custom_model_config["all_embedding_size"]
        self.rnn_size = lstm_state_size

        # bs * obs
        self.obs_encoder = nn.Linear(self.obs_size, self.obs_embedding_size)

        # bs * N * entity_shape
        self.entity_encoder_q = nn.Linear(self.entity_shape, self.entity_embedding_size)
        self.entity_encoder_k = nn.Linear(self.entity_shape, self.entity_embedding_size) 
        self.entity_encoder_v = nn.Linear(self.entity_shape, self.entity_embedding_size)

        self.attention = ScaleDotProductionAttention()
        self.all_encoder = nn.Linear(self.obs_embedding_size + self.entity_embedding_size, self.all_embedding_size)
        self.rnn = nn.LSTM(input_size=self.all_embedding_size, hidden_size=self.rnn_size, batch_first=True)
        # self.logits = nn.Linear(self.rnn_size, self.action_shape)
        # self.values = nn.Linear(self.rnn_size, 1)
        self.logits= nn.Sequential(
            nn.Linear(self.rnn_size, 256),
            nn.ReLU(),
            nn.Linear(256, 36),
        )
        self.values = nn.Sequential(
            nn.Linear(self.rnn_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
        )
        

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.obs_encoder.weight.new(1, self.rnn_size).zero_().squeeze(0),
            self.obs_encoder.weight.new(1, self.rnn_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.values(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        obs = inputs[:, :, :50]
        bs = inputs.shape[0]
        seq = inputs.shape[1]
        entities = inputs[:,:,50:].reshape(-1, seq, self.entity_shape)
        mask = entities.sum(axis=-1)

        obs_embedding = F.relu(self.obs_encoder(obs))

        q, k, v = [self.entity_encoder_q(entities), self.entity_encoder_k(entities), self.entity_encoder_v(entities)]
        q = q.reshape(bs*seq, -1, self.entity_embedding_size)
        k = k.reshape(bs*seq, -1, self.entity_embedding_size)
        v = v.reshape(bs*seq, -1, self.entity_embedding_size)
        mask = mask.reshape(bs*seq, -1, 1)
        mask_attn = torch.matmul(mask, mask.permute(0, 2, 1))

        entity_embeddings = self.attention(q, k, v, mask=mask_attn.detach())

        entity_embedding = torch.mean(entity_embeddings, dim=1, keepdim=True)
        entity_embedding = entity_embedding.reshape(bs, seq, self.entity_embedding_size)
        # import pdb; pdb.set_trace()
        all_embedding = torch.cat((obs_embedding, entity_embedding), dim=-1)
        core = F.relu(self.all_encoder(all_embedding))
        
        # packed_input = pack_padded_sequence(output, self.sequence_length)
        self._features, [h,c] = self.rnn(core, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        logits = self.logits(self._features)
        return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]




