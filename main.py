import random
import numpy as np
import time
import torch
from torchvision import datasets, transforms, utils
from models.Nets import CNNMnist
from options import args_parser
from client import *
from server import *
import copy
from termcolor import colored
import matplotlib.pyplot as plt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set the seed
set_seed(42)

def load_dataset():
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
    return dataset_train, dataset_test

def create_client_server(dataset_train, args):
    num_items = int(len(dataset_train) / args.num_users)
    clients, all_idxs = [], [i for i in range(len(dataset_train))]
    net_glob = CNNMnist(args=args).to(args.device)

    # divide training data, i.i.d.
    for i in range(args.num_users):
        new_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - new_idxs)
        new_client = Client(args=args, dataset=dataset_train, idxs=new_idxs, w=copy.deepcopy(net_glob.state_dict()))
        clients.append(new_client)

    server = Server(args=args, w=copy.deepcopy(net_glob.state_dict()))

    return clients, server

if __name__ == '__main__':

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args.device)

    print("load dataset...")
    dataset_train, dataset_test = load_dataset()

    print("clients and server initialization...")
    clients, server = create_client_server(dataset_train, args)

    # statistics for plot
    all_acc_train = []
    all_acc_test = []
    all_loss_glob = []

    # Evaluate initial model before any training
    print("Initial evaluation before any training")
    acc_train, _ = server.test(dataset_train)
    acc_test, _ = server.test(dataset_test)
    print("Initial Train Accuracy: {:.2f}%".format(acc_train))
    print("Initial Test Accuracy: {:.2f}%".format(acc_test))

    # training
    print("start training...")
    print('Algorithm:', colored(args.mode, 'green'))

    for iter in range(args.epochs):
        epoch_start = time.time()

        server.clients_update_w, server.clients_loss = [], []
        for idx in range(args.num_users):
            update_w, loss = clients[idx].train()
            server.clients_update_w.append(update_w)
            server.clients_loss.append(loss)

        # calculate global weights
        w_glob, loss_glob = server.FedAvg()

        # update local weights
        for idx in range(args.num_users):
            clients[idx].update(w_glob)

        epoch_end = time.time()
        print(colored('=====Epoch {:3d}====='.format(iter), 'yellow'))
        print('Training time:', epoch_end - epoch_start)

        if args.mode in ['Paillier', 'DP_Paillier']:
            server.model.load_state_dict(copy.deepcopy(clients[0].model.state_dict()))

        # testing
        acc_train, loss_train = server.test(dataset_train)
        acc_test, loss_test = server.test(dataset_test)
        print("Training accuracy: {:.2f}".format(acc_train))
        print("Testing accuracy: {:.2f}".format(acc_test))
        print('Training average loss {:.3f}'.format(loss_glob))
        all_acc_train.append(acc_train)
        all_acc_test.append(acc_test)
        all_loss_glob.append(loss_glob)

    # Final output for dp_tune.py regex
    print("Train Accuracy: {:.2f}, Test Accuracy: {:.2f}, Loss: {:.3f}".format(
        all_acc_train[-1], all_acc_test[-1], all_loss_glob[-1]
    ))

    # plot learning curve
    if not args.no_plot:
        x = np.linspace(0, args.epochs - 1, args.epochs)
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        plt.suptitle('Learning curves of ' + args.mode)
        ax1.plot(x, all_acc_train)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train accuracy')
        ax2.plot(x, all_acc_test)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Testing accuracy')
        ax3.plot(x, all_loss_glob)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Training average loss')
        plt.savefig('figs/' + args.mode + '_training_curve.png')
        plt.show()
