#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # experiment arguments
    parser.add_argument(
        '--mode', 
        type=str, 
        default='plain', 
        choices=['plain', 'DP', 'Paillier', 'DP_Paillier'], 
        help="plain: no privacy; DP: Differential Privacy; Paillier: Homomorphic Encryption; DP_Paillier: combined"
    )

    # federated arguments
    parser.add_argument('--epochs', type=int, default=6, help="number of global training rounds")
    parser.add_argument('--num_users', type=int, default=5, help="number of users (clients)")
    parser.add_argument('--local_ep', type=int, default=3, help="number of local training epochs")
    parser.add_argument('--local_bs', type=int, default=64, help="local training batch size")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.015, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum")
    parser.add_argument('--split', type=str, default='user', choices=['user', 'sample'], help="train-test split type")

    # Differential Privacy (DP) arguments
    parser.add_argument('--C', type=float, default=0.5, help="Gradient norm clip value for DP")
    parser.add_argument('--sigma', type=float, default=0.05, help="Gaussian noise std dev for DP")

    # other arguments
    parser.add_argument('--num_classes', type=int, default=10, help="number of output classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of image channels")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, use -1 for CPU")
    parser.add_argument('--no-plot', action="store_true", default=False, help="Disable learning curve plotting")

    args = parser.parse_args()
    return args
