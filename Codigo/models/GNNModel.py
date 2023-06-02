from abc import ABC, abstractmethod
import numpy as np

from dgllife.model.model_zoo.gatv2_predictor import GATv2Predictor
from dgllife.model.model_zoo import AttentiveFPPredictor, MGCNPredictor, GINPredictor


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
        return loss.item()

    def eval_step(self, reg, bg, labels, masks, loss_criterion, transformer):
        """ Compute loss_criterion and the absolute error after undoing the transformation from transformer """
        prediction = self.get_predictions(bg)
        loss = (loss_criterion(prediction, labels, reduction='none') * (masks != 0).float()).mean().item()

        prediction = prediction.cpu().numpy().reshape(-1, 1)
        labels = labels.cpu().numpy().reshape(-1, 1)
        if transformer is not None:
            abs_errors = np.abs(
                transformer.inverse_transform(prediction) - transformer.inverse_transform(labels)
            )
        else:
            abs_errors = np.abs(prediction - labels)
        return loss, abs_errors

    def get_predictions(self, bg):
        return self._model(bg, bg.ndata['h'])


class GATv2Model(GNNModel):
    def __init__(self, *, in_feats, agg_modes, hidden_feats, allow_zero_in_degree, **kwargs):
        super().__init__(in_feats=in_feats, agg_modes=agg_modes, hidden_feats=hidden_feats, allow_zero_in_degree=allow_zero_in_degree, **kwargs)
    def _init_model(self):
        self._model = GATv2Predictor(
            in_feats=self.in_feats, agg_modes=self.agg_modes, hidden_feats=self.hidden_feats, allow_zero_in_degree=self.allow_zero_in_degree
        )
    def get_predictions(self, bg):
        return self._model(bg, bg.ndata['h'])

class AttentiveFPModel(GNNModel):
    def __init__(self, *, node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout, **kwargs):
        super().__init__(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size, num_layers=num_layers, graph_feat_size=graph_feat_size, dropout=dropout, **kwargs)
    def _init_model(self):
        self._model = AttentiveFPPredictor(
            node_feat_size=self.node_feat_size, edge_feat_size=self.edge_feat_size, num_layers=self.num_layers, graph_feat_size=self.graph_feat_size, dropout=self.dropout
        )
    def get_predictions(self, bg):
        return self._model(bg, bg.ndata['h'], bg.edata['e'])

class MGCNModel(GNNModel):
    def __init__(self, *, feats, n_layers, classifier_hidden_feats, num_node_types, num_edge_types, predictor_hidden_feats, **kwargs):
        super().__init__(feats=feats, n_layers=n_layers, classifier_hidden_feats=classifier_hidden_feats, num_node_types=num_node_types, num_edge_types=num_edge_types, predictor_hidden_feats=predictor_hidden_feats, **kwargs)
    def _init_model(self):
        self._model = MGCNPredictor(
            feats=self.feats, n_layers=self.n_layers, classifier_hidden_feats=self.classifier_hidden_feats, num_node_types=self.num_node_types, num_edge_types=self.num_edge_types, predictor_hidden_feats=self.predictor_hidden_feats
        )
    def get_predictions(self, bg):
        #Featurizer se realiza con otro fichero (alchemy.py)
        node_types = bg.ndata['h']
        distances = bg.edata['e']
        return self._model(bg, node_types, distances)

class GINModel(GNNModel):
    def __init__(self, *, num_node_emb_list, num_edge_emb_list, num_layers, emb_dim, dropout, readout, **kwargs):
        super().__init__(num_node_emb_list=num_node_emb_list, num_edge_emb_list=num_edge_emb_list, num_layers=num_layers, emb_dim=emb_dim, dropout=dropout, readout=readout, **kwargs)
    def _init_model(self):
        self._model = GINPredictor(
            num_node_emb_list=self.num_node_emb_list, num_edge_emb_list=self.num_edge_emb_list, num_layers=self.num_layers, emb_dim=self.emb_dim, dropout=self.dropout, readout=self.readout
        )
    def get_predictions(self, bg):
        categorical_node_feats = [bg.ndata.pop('atomic_number'), bg.ndata.pop('chirality_type')]
        categorical_edge_feats = [bg.edata.pop('bond_type'), bg.edata.pop('bond_direction_type')]
        return self._model(bg, categorical_node_feats, categorical_edge_feats)