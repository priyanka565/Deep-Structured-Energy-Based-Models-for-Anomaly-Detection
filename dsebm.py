import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.dense1 = nn.Linear(120, 128)
        self.dense2 = nn.Linear(128, 512)
        self.inv_dense2 = nn.Linear(512, 128)
        self.inv_dense1 = nn.Linear(128, 120)
        self.activation = nn.Softplus()

    def forward(self, x):
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.activation(self.inv_dense2(x))
        x = self.activation(self.inv_dense1(x))
        return x

class DSEBM(nn.Module):
    def __init__(self, opts):
        super(DSEBM, self).__init__()
        self.config = opts
        self.network = Network()
        self.b_prime = nn.Parameter(torch.zeros(opts['batch_size'], 120))
        self.to(device)  # Move the entire model to the device

    def forward(self, x_input, is_training=True):
        x_input = x_input.requires_grad_(True)
        if is_training:
            noise = torch.randn_like(x_input)
            x_noise = x_input + noise
        else:
            x_noise = x_input

        net_out = self.network(x_input)
        net_noise_out = self.network(x_noise)

        energy = 0.5 * torch.sum((x_input - self.b_prime) ** 2) - torch.sum(net_out)
        energy_noise = 0.5 * torch.sum((x_noise - self.b_prime) ** 2) - torch.sum(net_noise_out)

        fx = x_input - torch.autograd.grad(energy, x_input, create_graph=True, allow_unused=True)[0]
        fx_noise = x_noise - torch.autograd.grad(energy_noise, x_noise, create_graph=True)[0]

        loss = torch.mean((x_input - fx_noise) ** 2)

        flat = (x_input - self.b_prime).view(x_input.size(0), -1)
        list_score_energy = 0.5 * torch.sum(flat ** 2, dim=1) - torch.sum(net_out, dim=1)

        delta = x_input - fx
        delta_flat = delta.view(delta.size(0), -1)
        list_score_recon = torch.norm(delta_flat, p=2, dim=1)

        return loss, list_score_energy, list_score_recon

    def train_model(self, data):
        opts = self.config
        optimizer = optim.Adam(self.parameters(), lr=opts['lr'])
        
        self.train()
        for epoch in range(opts['epoch_num']):
            sum_loss = 0
            batch_num = data.train_data.shape[0] // opts['batch_size']
            
            for _ in range(batch_num):
                batch_index = np.random.choice(data.train_data.shape[0], opts['batch_size'], replace=False)
                batch_data = torch.FloatTensor(data.train_data[batch_index]).to(device)
                
                optimizer.zero_grad()
                loss, _, _ = self(batch_data)
                loss.backward()
                optimizer.step()
                
                sum_loss += loss.item()
            
            print(f"Epoch {epoch}, Loss {sum_loss/batch_num}")
            self.evaluate(data)

    def evaluate(self, data):
        self.eval()
        opts = self.config
        num_test_points = data.test_data.shape[0]
        batch_size = opts['batch_size']
        batch_num = num_test_points // batch_size
        
        energy_score = []
        recon_score = []
        true_label = []
        
        for _ in range(batch_num):
            batch_index = np.random.choice(num_test_points, batch_size, replace=False)
            batch_data = torch.FloatTensor(data.test_data[batch_index]).to(device)
            batch_label = data.test_label[batch_index]
            
            _, score_e, score_r = self(batch_data, is_training=False)
            
            energy_score.extend(score_e.detach().cpu().numpy())
            recon_score.extend(score_r.detach().cpu().numpy())
            true_label.extend(batch_label)
        
        energy_score = np.array(energy_score)
        recon_score = np.array(recon_score)
        true_label = np.array(true_label)
        
        print("DSEBM-e:")
        self.compute_score(energy_score, true_label)
        print("DSEBM-r:")
        self.compute_score(recon_score, true_label)

    @staticmethod
    def compute_score(score_list, labels):
        num_test_points = labels.shape[0]
        score_sort_index = np.argsort(score_list)
        y_pred = np.zeros_like(labels)
        y_pred[score_sort_index[-int(num_test_points * 0.2):]] = 1
        precision, recall, f1, _ = precision_recall_fscore_support(labels.astype(int),
                                                                   y_pred.astype(int),
                                                                   average='binary')
        print(f"precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
