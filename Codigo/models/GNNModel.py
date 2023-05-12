from abc import ABC, abstractmethod

from dgllife.model.model_zoo.gatv2_predictor import GATv2Predictor


# ABC is to define an abstract class
class GNNModel(ABC):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._init_model()

    @abstractmethod
    def _init_model(self):
        pass

    def train_step(self, reg, bg, labels, masks, loss_criterion, optimizer):
        optimizer.zero_grad()
        prediction = self.get_predictions(bg)
        loss = (loss_criterion(prediction, labels, reduction='none') * (masks != 0).float()).mean()
        loss.backward()
        optimizer.step()
        return loss.data.item()

    def get_predictions(self, bg):
        return self._model(bg, bg.ndata['h'], bg.ndata['e'])


class GATv2Model(GNNModel):
    def __init__(self, *, in_feats, agg_modes, hidden_feats, allow_zero_in_degree, **kwargs):
        super().__init__(in_feats=in_feats, agg_modes=agg_modes, hidden_feats=hidden_feats, allow_zero_in_degree=allow_zero_in_degree, **kwargs)
    def _init_model(self):
        self._model = GATv2Predictor(
            in_feats=self.in_feats, agg_modes=self.agg_modes, hidden_feats=self.hidden_feats, allow_zero_in_degree=self.allow_zero_in_degree
        )
    def get_predictions(self, bg):
        return self._model(bg, bg.ndata['h'])
