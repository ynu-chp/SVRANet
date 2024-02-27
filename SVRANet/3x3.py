import torch
import logging
from time import time
import datetime
from trainer import Trainer
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--data_dir', type=str, default='data/3x3')
    parser.add_argument('--training_set', type=str, default='train')
    parser.add_argument('--test_set', type=str, default='test')


    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--d_hidden', type=int, default=64)

    parser.add_argument('--n_bidder', type=int, default=3)#user
    parser.add_argument('--n_item', type=int, default=3)#es
    parser.add_argument('--r_train', type=int, default=25, help='Number of steps in the inner maximization loop')
    parser.add_argument('--r_test', type=int, default=200, help='Number of steps in the inner maximization loop when testing')
    parser.add_argument('--gamma', type=float, default=1e-3, help='The learning rate for the inner maximization loop')


    parser.add_argument('--budget', type=int, default=0.5)
    parser.add_argument('--n_misreport_init', type=int, default=100)
    parser.add_argument('--n_misreport_init_train', type=int, default=1)

    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--misreport_epoch', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--test_size', type=float, default=None)
    parser.add_argument('--batch_test', type=int, default=50)
    parser.add_argument('--eval_freq', type=int, default=20)

    #rgt
    parser.add_argument('--lamb_rgt', type=float, default=1)
    parser.add_argument('--lamb_rgt_update_freq', type=int, default=6)
    parser.add_argument('--rho_rgt', type=float, default=1)#初始惩罚项ρ=1
    parser.add_argument('--rho_rgt_update_freq', type=int, default=2)
    parser.add_argument('--delta_rho_rgt', type=float, default=1)#ρ每次增大3
    #ir
    parser.add_argument('--lamb_ir', type=float, default=1)
    parser.add_argument('--lamb_ir_update_freq', type=int, default=6)
    parser.add_argument('--rho_ir', type=float, default=1)
    parser.add_argument('--rho_ir_update_freq', type=int, default=2)
    parser.add_argument('--delta_rho_ir', type=float, default=1)
    #bf
    parser.add_argument('--lamb_bf', type=float, default=1)
    parser.add_argument('--lamb_bf_update_freq', type=int, default=6)
    parser.add_argument('--rho_bf', type=float, default=1)
    parser.add_argument('--rho_bf_update_freq', type=int, default=2)
    parser.add_argument('--delta_rho_bf', type=float, default=1)
    #bw
    parser.add_argument('--lamb_bw', type=float, default=1)
    parser.add_argument('--lamb_bw_update_freq', type=int, default=6)
    parser.add_argument('--rho_bw', type=float, default=1)
    parser.add_argument('--rho_bw_update_freq', type=int, default=2)
    parser.add_argument('--delta_rho_bw', type=float, default=1)
    #thp
    parser.add_argument('--lamb_thp', type=float, default=1)
    parser.add_argument('--lamb_thp_update_freq', type=int, default=6)
    parser.add_argument('--rho_thp', type=float, default=1)
    parser.add_argument('--rho_thp_update_freq', type=int, default=2)
    parser.add_argument('--delta_rho_thp', type=float, default=1)

    t0 = time()
    args = parser.parse_args()
    trainer = Trainer(args)

    trainer.train(args)

    time_used = time() - t0
    logging.info(f'Time Cost={datetime.timedelta(seconds=time_used)}')

