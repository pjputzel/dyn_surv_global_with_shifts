
from models.GlobalPlusEpsModel import GlobalPlusEpsModel
from models.BasicModelTrainer import BasicModelTrainer
from models.BasicModelThetaPerStep import BasicModelThetaPerStep
from models.DeltaIJModel import DeltaIJModel
from models.LinearDeltaIJModel import LinearDeltaIJModel
from models.LinearThetaIJModel import LinearThetaIJModel
from models.LinearDeltaIJModelNumVisitsOnly import LinearDeltaIJModelNumVisitsOnly
from models.DummyGlobalModel import DummyGlobalModel
from models.ConstantDeltaModelLinearRegression import ConstantDeltaModelLinearRegression
from models.EmbeddingConstantDeltaModelLinearRegression import EmbeddingConstantDeltaModelLinearRegression

# put models_dictionary here

models_dict = \
    {
        'theta_per_step': BasicModelThetaPerStep,

    }
