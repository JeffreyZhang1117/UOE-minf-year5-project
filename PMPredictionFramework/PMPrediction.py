from abc import abstractmethod
from PreProcessing.StaticPersonalPreprocessHelper import *
from PreProcessing.StaticOnlyPreprocessHelper import *
from Spatial.SpatialPredictions import *
from Spatial.SpatialPredictionsSS import *
from Temporal.TemporalPredictions import *
from Temporal.TemporalPredictionsSS import *
from SpatioTemporal.SpatioTemporalPredictions import *
from SpatioTemporal.SpatioTemporalPredictionsSS import *
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

class PMPredictionAlgorithm:
    def __init__(self, model, dataset, spatialChoice, temporalChoice, spatioTemporalChoice):
        self.model = model
        self.dataset = dataset
        self.spatialChoice = spatialChoice
        self.temporalChoice = temporalChoice
        self.spatioTemporalChoice = spatioTemporalChoice


    def preprocess(self):
        print('Parent Preprocessor')

    @abstractmethod
    def predictSpatial(self):
        pass

    @abstractmethod
    def predictTemporal(self):
        pass

    @abstractmethod
    def predictSpatioTemporal(self):
        pass

    def mean_absolute_percentage_error_percentage(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def evaluate(self, trueValues, predictedValues):
        mae = mean_absolute_error(np.asarray(trueValues), predictedValues)

        mape = mean_absolute_percentage_error(list(trueValues), predictedValues)
        mape_percentage = mean_absolute_percentage_error_percentage(list(trueValues), predictedValues)
        print(mae, mape, mape_percentage)
        return mae, mape

class StaticAndPersonal(PMPredictionAlgorithm):
    def preprocess(self, statusMsg, root):
        preprocess(self.dataset)

    def predictSpatial(self, statusMsg, root):
        return predictSpatialValues(self.dataset, self.spatialChoice)

    def predictTemporal(self, statusMsg, root):
        return predictTemporalValues(self.dataset, self.temporalChoice)

    def predictSpatioTemporal(self):
        return predictSpatioTemporalValues(self.dataset, self.spatioTemporalChoice)

class StaticOnly(PMPredictionAlgorithm):

    def preprocess(self, statusMsg, root):
        preprocessSS(self.dataset)

    def predictSpatial(self, statusMsg, root):
        return predictSpatialValuesSS(self.dataset, self.spatialChoice)

    def predictTemporal(self, statusMsg, root):
        return predictTemporalValuesSS(self.dataset, self.temporalChoice)

    def predictSpatioTemporal(self):
        return predictSpatioTemporalValuesSS(self.dataset, self.spatioTemporalChoice)
