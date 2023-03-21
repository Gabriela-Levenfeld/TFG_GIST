import torch
from sklearn.model_selection import train_test_split

from utils.data import load_alvadesc_data, load_mols_df
from utils.train.torchutils import *
import utils.train.model_selection as model_selection
from utils.train.loss import *
from sklearn import metrics

if __name__ == "__main__":
    # Use this variable to differentiate fast test (= smoke tests) Vs "serious" tests
    SMOKE = True

    n = 10 if SMOKE else None
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



    # Task 2) Graph nnet
    # X, y = load_mols_df()
    # print(X.shape)
    # print(y.shape)
    # # TODO: transform mols to graphs using dgllife.utilsmol_to_bigraph
