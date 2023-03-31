import dgl.nn.pytorch
import dgllife.utils
import torch
from sklearn.model_selection import train_test_split

from utils.data import load_alvadesc_data, load_mols_df
from utils.train.torchutils import *
import utils.train.model_selection as model_selection
from utils.train.loss import *
from sklearn import metrics

from dgllife.utils import mol_to_bigraph, ScaffoldSplitter
from utils.featurizers import get_atom_featurizer, get_bond_featurizer
from rdkit import Chem

if __name__ == "__main__":
    # Use this variable to differentiate fast test (= smoke tests) Vs "serious" tests
    SMOKE = True

    n = 10 if SMOKE else None

    # Task 2) Graph nnet
    X, y = load_mols_df()
    print(X.shape)
    print(y.shape)

    # # TODO: transform mols to graphs using dgllife.utils.mol_to_bigraph
    def featurize_atoms (mol):
        feats = []
        for atom in mol.GetAtoms():
            feats.append(atom.GetAtomicNum())
        return {'atomic': torch.tensor(feats).reshape(-1, 1).float()}

    def featurize_bonds(mol):
        feats = []
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        for bond in mol.GetBonds():
            btype = bond_types.index(bond.GetBondType())
            # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
            feats.extend([btype, btype])
        return {'type': torch.tensor(feats).reshape(-1, 1).float()}


    def atom_featurizer(atom):
        return torch.nn.functional.one_hot(torch.tensor([atom.GetAtomicNum() - 1]), num_classes=118)


    # mol -> DGL graph
    graph_list = []
    for molecule in X[80033:]:
        print(molecule.GetBonds()[0].GetBondType())
        graph_X = mol_to_bigraph(molecule, add_self_loop=True, node_featurizer=featurize_atoms,
                                 edge_featurizer=featurize_bonds)
        #graph_X = mol_to_bigraph(molecule)
        graph_list.append(graph_X)

    print(graph_list)
    batched_graph = dgl.batch(graph_list) #Convert the graph into a tensor
    print('Batches')
    print(batched_graph)

    #node_feats = batched_graph.ndata.pop('h')
    #edge_feats = batched_graph.edata.pop('e')
    labels = torch.tensor(y)


    # Define the Model
    class GraphNNnet(torch.nn.Module):
        def __init__(self, in_feats, hidden_feats, out_feats):
            super(GraphNNnet, self).__init__()

            self.conv1 = dgl.nn.GraphConv(in_feats, hidden_feats)
            self.conv2 = dgl.nn.GraphConv(hidden_feats, out_feats)

            #self.dropout = torch.nn.Dropuot(dropout)

        def forward(self, g, feats):
            h = self.conv1(g, feats)
            #h = self.dropuot(h)
            h = torch.nn.ReLu(h)
            h = self.conv2(g, h)
            return h



    #Split data: training and test
    #train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(X, frac_train=0.8, frac_val=0.1, frac_test=0.1, scaffold_func='smiles')
    #Training
    #Testing










    """
    # Task 1) Nnet with "standard" features
    X, y = load_alvadesc_data(n=n)
    print(X.shape)
    print(y.shape)

    #Define the Model
    class LinearRegression(torch.nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.layer1 = torch.nn.Linear(2214,1107)
            self.relu = torch.nn.ReLU()
            self.layer2 = torch.nn.Linear(1107, 1)

        def forward(self, features):
            x = self.layer1(features)
            x = self.relu(x)
            x = self.layer2(x)
            return x

    #Data (numpy.ndarray) -> Tensor
    device = get_default_device()
    X_tensor = to_torch(X, device)
    y_tensor = to_torch(y, device)

    #Split data: training and test -> EN EL TEST_SIZE NO ME DEJA UTILIZAR VALORES MENORES A 0.5
    X_train, X_test, y_train, y_test = model_selection.stratified_train_test_split(X_tensor, y_tensor.squeeze(), test_size=0.5)

    net = LinearRegression()
    loss_function = torch.nn.MSELoss()
    optimizer_function = torch.optim.SGD(net.parameters(), lr=0.01)

    print('Training NN')
    #Training the NN
    for i in range(100):
        output = net(X_train) #Forward propagation
        loss = loss_function(output, y_train.unsqueeze(1)) #Compute loss
        #loss = truncated_rmse(output.detach().numpy(), y_train) HE INTENTADO USAR SU FUNCION PERO NO ME FUNCIONA
        net.zero_grad() #Zero out the gradients
        loss.backward() #Backpropagation
        optimizer_function.step() #Update
        print(f'{i} - {loss.item()}')

    print('Testing NN')
    #Testing the NN
    with torch.no_grad():
        y_pred = net(X_test)
        test_loss = loss_function(y_pred, y_test.unsqueeze(1))
    print(f'Test loss: {test_loss:.4f}')
    """
