REGISTRY = {}

from .rnn_agent import RNNAgent
from .task_encoder_rnn_agent import TaskEncoderRNNAgent, AuxiliaryTaskRNNAgent, ContrastiveRNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .rnn_sd_agent import RNN_SD_Agent

REGISTRY["rnn"] = RNNAgent
#REGISTRY["task_encoder_rnn"] = TaskEncoderRNNAgent
#REGISTRY["availableA_prediction_rnn"] = AuxiliaryTaskRNNAgent
#REGISTRY["availableA_contrastive_rnn"] = ContrastiveRNNAgent
#REGISTRY["rnn_ns"] = RNNNSAgent
#REGISTRY["rnn_feat"] = RNNFeatureAgent
#REGISTRY["rnn_sd"] = RNN_SD_Agent
