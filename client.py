import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from models.Nets import CNNMnist
import copy
import time
from phe import paillier  # Make sure this is present for encryption

# Constants
SCALING_FACTOR = 1e5  # Used for quantization of model updates before encryption

global_pub_key, global_priv_key = paillier.generate_paillier_keypair()

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Client():
    
    def __init__(self, args, dataset=None, idxs=None, w = None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.model = CNNMnist(args=args).to(args.device)
        self.model.load_state_dict(w)
        # DP hyperparameter
        self.C = self.args.C
        # Paillier keys for encryption/decryption
        if self.args.mode in ['Paillier', 'DP_Paillier']:
            self.pub_key = global_pub_key
            self.priv_key = global_priv_key
        
    def train(self):
        w_old = copy.deepcopy(self.model.state_dict())
        net = copy.deepcopy(self.model)

        # train and update
        net.train()   
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        
        w_new = net.state_dict()
        update_w = {}

        # PLAIN
        if self.args.mode == 'plain':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]

        # DIFFERENTIAL PRIVACY
        elif self.args.mode == 'DP':
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                sensitivity = torch.norm(update_w[k], p=2)
                update_w[k] = update_w[k] / max(1, sensitivity / self.C)

        # PAILLIER ENCRYPTION
        elif self.args.mode == 'Paillier':
            print('encrypting...')
            enc_start = time.time()
            for k in w_new.keys():
                update_w[k] = w_new[k] - w_old[k]
                list_w = update_w[k].view(-1).cpu().tolist()
                update_w[k] = [self.pub_key.encrypt(elem) for elem in list_w]
            enc_end = time.time()
            print('Encryption time:', enc_end - enc_start)

        # DP + PAILLIER
        elif self.args.mode == 'DP_Paillier':
            print('Applying DP and encrypting...')
            enc_start = time.time()
            for k in w_new.keys():
                # Compute the update
                update = w_new[k] - w_old[k]

                # Clip the update
                norm = torch.norm(update, p=2)
                clip_coef = self.C / (norm + 1e-6)
                if clip_coef < 1:
                    update = update * clip_coef

                # Add Gaussian noise
                noise = torch.normal(0, self.args.sigma, size=update.shape).to(self.args.device)
                update += noise

                # Quantize the update
                update = (update * SCALING_FACTOR).to(torch.int64)

                # Flatten and convert to list
                update_list = update.view(-1).tolist()

                # Encrypt the update
                update_w[k] = [self.pub_key.encrypt(int(val)) for val in update_list]
            enc_end = time.time()
            print('Encryption time:', enc_end - enc_start)


        else:
            raise NotImplementedError

        return update_w, sum(batch_loss) / len(batch_loss)

    def update(self, w_glob):
        if self.args.mode in ['plain', 'DP']:
            self.model.load_state_dict(w_glob)

        elif self.args.mode == 'DP_Paillier':
            update_w_avg = copy.deepcopy(w_glob)
            print('Decrypting...')
            dec_start = time.time()
            for k in update_w_avg.keys():
                # Decrypt the update
                decrypted = [self.priv_key.decrypt(val) for val in update_w_avg[k]]

                # Convert to tensor and reshape
                decrypted_tensor = torch.tensor(decrypted, dtype=torch.float32).to(self.args.device)
                decrypted_tensor = decrypted_tensor.view(self.model.state_dict()[k].shape)

                # Dequantize the update
                decrypted_tensor = decrypted_tensor / SCALING_FACTOR

                # Update the model parameters
                self.model.state_dict()[k] += decrypted_tensor
            dec_end = time.time()
            print('Decryption time:', dec_end - dec_start)

        else:
            raise NotImplementedError
