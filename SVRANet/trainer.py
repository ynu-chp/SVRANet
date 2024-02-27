import os
import sys
import numpy as np
from statistics import mean
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json
import shutil
from time import time

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
datefmt="%Y-%m-%d,%H:%M:%S",
)

from network import TransformerMechanism
from utilities import misreportOptimization
def loss_function(mechanism,lamb_rgt,rho_rgt,lamb_ir,rho_ir,lamb_bf,rho_bf,lamb_bw,rho_bw,lamb_thp,rho_thp,batch,trueValuations,misreports,budget):
    from utilities import loss
    return loss(mechanism,lamb_rgt,rho_rgt,lamb_ir,rho_ir,lamb_bf,rho_bf,lamb_bw,rho_bw,lamb_thp,rho_thp,batch,trueValuations,misreports,budget)


class Trainer():
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.set_data(args)
        self.set_model(args)

        self.rho_rgt = args.rho_rgt
        self.lamb_rgt = args.lamb_rgt * torch.ones(args.n_bidder).to(args.device)

        self.rho_ir=args.rho_ir
        self.lamb_ir=args.lamb_ir*torch.ones(args.n_bidder).to(args.device)

        self.rho_bf = args.rho_bf
        self.lamb_bf = args.lamb_bf * torch.ones(1).to(args.device)

        self.rho_bw = args.rho_bw
        self.lamb_bw = args.lamb_bf * torch.ones(args.n_bidder,args.n_item).to(args.device)

        self.rho_thp = args.rho_thp
        self.lamb_thp = args.lamb_thp * torch.ones(args.n_item).to(args.device)

        self.budget=args.budget
        self.n_iter = 0

    def set_data(self, args):
        def load_data(dir):
            data = [np.load(os.path.join(dir, 'bid.npy')).astype(np.float32),
                    np.load(os.path.join(dir, 'value.npy')).astype(np.float32),
                    np.load(os.path.join(dir, 'bw.npy')).astype(np.float32),
                    np.load(os.path.join(dir, 'bwr.npy')).astype(np.float32),
                    np.load(os.path.join(dir, 'thp.npy')).astype(np.float32)]
            return tuple(data)

        self.train_dir = os.path.join(args.data_dir, args.training_set)
        self.train_data = load_data(self.train_dir)
        self.train_size = len(self.train_data[0])

        self.misreports = np.random.uniform(0,1,size=(self.train_size, args.n_misreport_init_train,args.n_bidder))

        self.test_dir = os.path.join(args.data_dir, args.test_set)
        self.test_data = load_data(self.test_dir)
        self.test_size = len(self.test_data[0])

    def set_model(self, args):
        self.mechanism = TransformerMechanism(args.n_layer, args.n_head, args.d_hidden).to(args.device)
        self.mechanism = nn.DataParallel(self.mechanism)#device_ids=[0, 1]
        # state_dict = torch.load('model/budget=1.6/4x2')
        # self.mechanism.load_state_dict(state_dict)
        self.optimizer = torch.optim.Adam(self.mechanism.parameters(), lr=args.learning_rate)


    def train(self, args):
        for epoch in range(args.n_epoch):
            loss_sum = 0
            profit_sum = 0
            rgt_sum = 0
            rgt_max = 0
            ir_sum=0
            ir_max=0
            bf_sum=0
            bf_max=0
            bw_sum=0
            bw_max=0
            thp_sum=0
            thp_max=0
            payment_sum=0

            for i in tqdm(range(0, self.train_size, args.batch_size)):
                self.n_iter += 1
                batch_indices = np.random.choice(self.train_size, args.batch_size)


                self.misreports = misreportOptimization(self.mechanism, batch_indices, self.train_data, self.misreports,
                                                   args.r_train, args.gamma)



                loss, rgt_mean_bidder, rgt_max_batch, ir_mean_bidder,ir_max_batch,bf_mean_batch,bf_max_batch,bw_mean_bidder_item,bw_max_batch,thp_mean_item,thp_max_batch,profit,payment = \
                    loss_function(self.mechanism, self.lamb_rgt, self.rho_rgt, self.lamb_ir, self.rho_ir, self.lamb_bf,self.rho_bf,self.lamb_bw,self.rho_bw,self.lamb_thp,self.rho_thp, batch_indices, self.train_data, self.misreports ,self.budget)

                loss_sum += loss.item() * len(batch_indices)

                rgt_sum += rgt_mean_bidder.mean().item() * len(batch_indices)
                rgt_max = max(rgt_max, rgt_max_batch.item())

                ir_sum += ir_mean_bidder.mean().item() * len(batch_indices)
                ir_max = max(ir_max, ir_max_batch.item())

                bf_sum += bf_mean_batch.mean().item()*len(batch_indices)
                bf_max=max(bf_max,bf_max_batch.item())

                bw_sum += bw_mean_bidder_item.mean().item() * len(batch_indices)
                bw_max = max(bw_max, bw_max_batch.item())

                thp_sum += thp_mean_item.mean().item() * len(batch_indices)
                thp_max = max(thp_max, thp_max_batch.item())

                profit_sum += profit.item() * len(batch_indices)

                payment_sum+=payment.item()*len(batch_indices)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


                if self.n_iter % args.lamb_rgt_update_freq  == 0:
                    self.lamb_rgt += self.rho_rgt * rgt_mean_bidder.detach()
                if self.n_iter % args.lamb_ir_update_freq  == 0:
                    self.lamb_ir += self.rho_ir * ir_mean_bidder.detach()
                if self.n_iter % args.lamb_bf_update_freq == 0:
                    self.lamb_bf += self.rho_bf * bf_mean_batch.detach()
                if self.n_iter % args.lamb_bw_update_freq == 0:
                    self.lamb_bw += self.rho_bw * bw_mean_bidder_item.detach()
                if self.n_iter % args.lamb_thp_update_freq == 0:
                    self.lamb_thp += self.rho_thp * thp_mean_item.detach()

            if (epoch + 1) % args.rho_rgt_update_freq == 0:
                self.rho_rgt += args.delta_rho_rgt
            if (epoch + 1) % args.rho_ir_update_freq == 0:
                self.rho_ir += args.delta_rho_ir
            if (epoch + 1) % args.rho_bf_update_freq == 0:
                self.rho_bf += args.delta_rho_bf
            if (epoch + 1) % args.rho_bw_update_freq == 0:
                self.rho_bw += args.delta_rho_bw
            if (epoch + 1) % args.rho_thp_update_freq == 0:
                self.rho_thp += args.delta_rho_thp

            logging.info(f"Train: epoch={epoch + 1}, loss={loss_sum/self.train_size}, "f"profit={(profit_sum)/self.train_size}")

            logging.info(f"rgt={(rgt_sum) / self.train_size}, rgt_max={rgt_max}")
            logging.info(f"irp={(ir_sum) / self.train_size}, irp_max={ir_max}")
            logging.info(f"bfp={(bf_sum) / self.train_size}, bfp_max={bf_max}")
            logging.info(f"bwp={(bw_sum) / self.train_size}, bwp_max={bw_max}")
            logging.info(f"thpp={(thp_sum) / self.train_size}, thpp_max={thp_max}")
            logging.info(f"payment={(payment_sum) / self.train_size}")

            logging.info(f"Train: rho_rgt={self.rho_rgt}, lamb_rgt={self.lamb_rgt.mean().item()} ")
            logging.info(f"Train: rho_irp={self.rho_ir}, lamb_irp={self.lamb_ir.mean().item()} ")
            logging.info(f"Train: rho_bfp={self.rho_bf}, lamb_bfp={self.lamb_bf.mean().item()} ")
            logging.info(f"Train: rho_bwp={self.rho_bw}, lamb_bwp={self.lamb_bw.mean().item()} ")
            logging.info(f"Train: rho_thpp={self.rho_thp}, lamb_thpp={self.lamb_thp.mean().item()} ")

            if (epoch + 1) % args.eval_freq== 0:
                self.test(args, valid=True)
        if args.n_epoch!=0:
            if not os.path.exists("model"):
                os.makedirs("model")
            if not os.path.exists(os.path.join("model", "budget="+str(args.budget))):
                os.makedirs(os.path.join("model", "budget="+str(args.budget)))

            model_path = os.path.join(os.path.join("model", "budget="+str(args.budget)), str(args.n_bidder) +"x"+ str(args.n_item))
            torch.save(self.mechanism.state_dict(), model_path)

        logging.info('Final test')
        self.test(args)


    def test(self, args, valid=False, load=False):
        if valid:
            data_size = args.batch_test * 10
            indices = np.random.choice(self.test_size, data_size)
            data = tuple([x[indices] for x in self.test_data])
        else:
            data_size = self.test_size
            data = self.test_data

        misreports = np.random.uniform(0,1, size=(data_size, args.n_misreport_init, args.n_bidder))


        indices = np.arange(data_size)
        loss_sum = 0
        profit_sum = 0
        rgt_sum = 0
        rgt_max = 0
        ir_sum = 0
        ir_max = 0
        bf_sum = 0
        bf_max = 0
        bw_sum = 0
        bw_max = 0
        thp_sum = 0
        thp_max = 0
        payment_sum=0
        n_iter = 0.0
        for i in tqdm(range(0, data_size, args.batch_test)):
            batch_indices = indices[i:i+args.batch_test]

            n_iter += len(batch_indices)
            misreports = misreportOptimization(self.mechanism, batch_indices, data, misreports,
                                               args.r_test, args.gamma)

            with torch.no_grad():
                loss, rgt_mean_bidder, rgt_max_batch, ir_mean_bidder, ir_max_batch, bf_mean_batch, bf_max_batch, bw_mean_bidder_item, bw_max_batch, thp_mean_item, thp_max_batch, profit, payment = \
                    loss_function(self.mechanism, self.lamb_rgt, self.rho_rgt, self.lamb_ir, self.rho_ir, self.lamb_bf,
                                  self.rho_bf, self.lamb_bw, self.rho_bw, self.lamb_thp, self.rho_thp, batch_indices,
                                  self.train_data, self.misreports, self.budget)

            loss_sum += loss.item() * len(batch_indices)

            rgt_sum += rgt_mean_bidder.mean().item() * len(batch_indices)
            rgt_max = max(rgt_max, rgt_max_batch.item())

            ir_sum += ir_mean_bidder.mean().item() * len(batch_indices)
            ir_max = max(ir_max, ir_max_batch.item())

            bf_sum += bf_mean_batch.mean().item() * len(batch_indices)
            bf_max = max(bf_max, bf_max_batch.item())

            bw_sum += bw_mean_bidder_item.mean().item() * len(batch_indices)
            bw_max = max(bw_max, bw_max_batch.item())

            thp_sum += thp_mean_item.mean().item() * len(batch_indices)
            thp_max = max(thp_max, thp_max_batch.item())

            profit_sum += profit.item() * len(batch_indices)
            payment_sum+=payment.item() * len(batch_indices)

            if valid == False:
                logging.info(f"profit={(profit_sum)/n_iter}, rgt={(rgt_sum)/n_iter}, irp={(ir_sum)/n_iter}, bfp={(bf_sum)/n_iter}, bwp={(bw_sum)/n_iter}, thpp={(thp_sum)/n_iter},payment={(payment_sum)/n_iter}")

        logging.info(f"Test: loss={loss_sum/data_size}, profit={(profit_sum)/data_size}, "
                     f"rgt={(rgt_sum)/data_size}, rgt_max={rgt_max}, "
                     f"irp={(ir_sum)/data_size}, irp_max={ir_max}, "
                     f"bfp={(bf_sum)/data_size},bfp_max={bf_max},"
                     f"bwp={(bw_sum) / data_size}, bwp_max={bw_max}, "
                     f"thpp={(thp_sum) / data_size},thpp_max={thp_max},"
                     f"payment={(payment_sum) / data_size}")






