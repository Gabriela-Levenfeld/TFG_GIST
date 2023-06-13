from abc import ABC, abstractmethod
import numpy as np

from dgllife.model.model_zoo.gatv2_predictor import GATv2Predictor
from dgllife.model.model_zoo import AttentiveFPPredictor, MPNNPredictor, GINPredictor


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
    def __init__(self, *, in_feats, hidden_feats, num_heads, feat_drops, attn_drops, alphas, residuals, allow_zero_in_degree, share_weights, agg_modes, predictor_out_feats, predictor_dropout, **kwargs):
        super().__init__(in_feats=in_feats, agg_modes=agg_modes, hidden_feats=hidden_feats, allow_zero_in_degree=allow_zero_in_degree, **kwargs)
    def _init_model(self):
        self._model = GATv2Predictor(
            in_feats=self.in_feats,
            hidden_feats=self.hidden_feats,
            num_heads=self.num_heads,
            feat_drops=self.feat_drops,
            attn_drops=self.attn_drops,
            alphas=self.alphas,
            residuals=self.residuals,
            allow_zero_in_degree=self.allow_zero_in_degree,
            share_weights=self.share_weights,
            agg_modes=self.agg_modes,
            predictor_out_feats=self.predictor_out_feats,
            predictor_dropout=self.predictor_dropout
        )
    def get_predictions(self, bg):
        feats = bg.ndata['h']
        return self._model(bg, feats)
    def train(self):
        self._model.train()
    def eval(self):
        self._model.eval()

class AttentiveFPModel(GNNModel):
    def __init__(self, *, node_feat_size, edge_feat_size, num_layers, graph_feat_size, dropout, **kwargs):
        super().__init__(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size, num_layers=num_layers, graph_feat_size=graph_feat_size, dropout=dropout, **kwargs)
    def _init_model(self):
        self._model = AttentiveFPPredictor(
            node_feat_size=self.node_feat_size,
            edge_feat_size=self.edge_feat_size,
            num_layers=self.num_layers,
            graph_feat_size=self.graph_feat_size,
            dropout=self.dropout
        )
    def get_predictions(self, bg):
        node_feats = bg.ndata['h']
        edge_feats = bg.edata['e']
        return self._model(bg, node_feats, edge_feats)
    def train(self):
        self._model.train()
    def eval(self):
        self._model.eval()

class MPNNModel(GNNModel):
    def __init__(self, *, node_in_feats, edge_in_feats, node_out_feats, edge_hidden_feats, **kwargs):
        super().__init__(node_in_feats=node_in_feats, edge_in_feats=edge_in_feats, node_out_feats=node_out_feats, edge_hidden_feats=edge_hidden_feats, **kwargs)
    def _init_model(self):
        self._model = MPNNPredictor(
            node_in_feats=self.node_in_feats,
            edge_in_feats=self.edge_in_feats,
            node_out_feats=self.node_out_feats,
            edge_hidden_feats=self.edge_hidden_feats
        )
    def get_predictions(self, bg):
        node_feats = bg.ndata['h']
        edge_feats = bg.edata['e']
        return self._model(bg, node_feats, edge_feats)
    def train(self):
        self._model.train()
    def eval(self):
        self._model.eval()

class GINModel(GNNModel):
    def __init__(self, *, num_node_emb_list, num_edge_emb_list, num_layers, emb_dim, dropout, readout, **kwargs):
        super().__init__(num_node_emb_list=num_node_emb_list, num_edge_emb_list=num_edge_emb_list, num_layers=num_layers, emb_dim=emb_dim, dropout=dropout, readout=readout, **kwargs)
    def _init_model(self):
        self._model = GINPredictor(
            num_node_emb_list=self.num_node_emb_list,
            num_edge_emb_list=self.num_edge_emb_list,
            num_layers=self.num_layers,
            emb_dim=self.emb_dim,
            dropout=self.dropout,
            readout=self.readout
        )
    def get_predictions(self, bg):
        categorical_node_feats = [bg.ndata.pop('atomic_number'), bg.ndata.pop('chirality_type')]
        categorical_edge_feats = [bg.edata.pop('bond_type'), bg.edata.pop('bond_direction_type')]
        return self._model(bg, categorical_node_feats, categorical_edge_feats)
    def train(self):
        self._model.train()
    def eval(self):
        self._model.eval()
