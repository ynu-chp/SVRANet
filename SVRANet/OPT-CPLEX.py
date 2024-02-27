from docplex.mp.model import Model
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import logging

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
datefmt="%Y-%m-%d,%H:%M:%S",
)
#不可信opt
def opt_utility(bid,value,bw,bwr,thp,budget):
    bidder,es=bwr.shape[0],bwr.shape[1]
    model=Model()
    allocation=model.continuous_var_list([i for i in range(bidder)],lb=0,ub=1,name='allocation')
    flow=[]
    for i in range(bidder):
        flow.append(model.continuous_var_list([j for j in range(es)],lb=0,name="flow_{}".format(i)))


    model.maximize(model.sum([allocation[i]*(value[i]-bid[i]) for i in range(bidder)]))#

    model.add_constraint(model.sum([bid[i]*allocation[i] for i in range(bidder)])<=budget)#

    for i in range(bidder):
            model.add_constraint(model.sum([flow[i][j] for j in range(es)])==bw[i]*allocation[i]) #

    for i in range(bidder):
        for j in range(es):
            model.add_constraint(flow[i][j]<=bwr[i][j])

    for i in range(es):
        model.add_constraint(model.sum([flow[j][i] for j in range(bidder)])<=thp[i])

    solution=model.solve()
    payment=0
    for i in range(bidder):
        payment+=solution[allocation[i]]*bid[i]
    # print("payment",payment)

    # print(solution)
    if(solution):
        return (solution.objective_value,payment)
    else:
        return (-1,-1)



es=3
for bidder in range(2,8):
    for budget in [k*0.5 for k in range(1,8)]:
        path = os.path.join(os.path.join("data", str(bidder) + "x" + str(es)), "train")

        bid = np.load(os.path.join(path, 'bid.npy')).astype(np.float32)
        value = np.load(os.path.join(path, 'value.npy')).astype(np.float32)
        bw  = np.load(os.path.join(path, 'bw.npy')).astype(np.float32)
        bwr = np.load(os.path.join(path, 'bwr.npy')).astype(np.float32)
        thp = np.load(os.path.join(path, 'thp.npy')).astype(np.float32)

        bs = bid.shape[0]
        total_utility = 0
        total_payment=0
        for i in tqdm(range(bs)):
            utility,payment=opt_utility(bid[i],value[i],bw[i],bwr[i],thp[i],budget)
            total_utility+=utility
            total_payment+=payment
        logging.info(f"budget={budget},bidder={bidder},es={es},opt-utility={total_utility/bs},opt-payment={total_payment/bs},opt-wf={(total_payment+total_utility)/bs}")
        print()
    print()

