
from models.GlobalPlusEpsModel import GlobalPlusEpsModel
from models.BasicModelTrainer import BasicModelTrainer
from models.BasicModelThetaPerStep import BasicModelThetaPerStep
from models.DeltaIJModel import DeltaIJModel
from models.LinearDeltaIJModel import LinearDeltaIJModel
from models.LinearThetaIJModel import LinearThetaIJModel
from models.LinearDeltaIJModelNumVisitsOnly import LinearDeltaIJModelNumVisitsOnly
from models.DummyGlobalModel import DummyGlobalModel
from models.RNNDeltaIJWithEmbedding import RNNDeltaIJWithEmbeddingModel
from models.ConstantDeltaModelLinearRegression import ConstantDeltaModelLinearRegression
from models.EmbeddingConstantDeltaModelLinearRegression import EmbeddingConstantDeltaModelLinearRegression

# put models_dictionary here

models_dict = \
    {
        'theta_per_step': BasicModelThetaPerStep,
        'linear_theta_per_step': LinearThetaIJModel,
        'dummy_global_zero_deltas': DummyGlobalModel,
        'dummy_global': DummyGlobalModel,
        'linear_delta_per_step': LinearDeltaIJModel,
        'linear_delta_per_step_num_visits_only': LinearDeltaIJModelNumVisitsOnly,
        'linear_constant_delta': ConstantDeltaModelLinearRegression,
        'embedding_linear_constant_delta': EmbeddingConstantDeltaModelLinearRegression,
        'RNN_delta_per_step': DeltaIJModel,
        'embedded_RNN_delta_per_step': RNNDeltaIJWithEmbeddingModel,
    }
