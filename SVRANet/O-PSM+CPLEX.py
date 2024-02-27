import numpy as np
import pandas as pd
import os
from docplex.mp.model import Model
from tqdm import tqdm
import logging

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s",
datefmt="%Y-%m-%d,%H:%M:%S",
)

def opt_flow(value,payment,bw,bwr,thp):
    bidder,es=bwr.shape[0],bwr.shape[1]
    model=Model()
    allocation=model.integer_var_list([i for i in range(bidder)],lb=0,ub=1,name='allocation')
    flow=[]
    for i in range(bidder):
        flow.append(model.continuous_var_list([j for j in range(es)],lb=0,name="flow_{}".format(i)))


    model.maximize(model.sum([allocation[i]*value[i] for i in range(bidder)]))

    # model.add_constraint(model.sum([bid[i]*allocation[i] for i in range(bidder)])<=budget)#budget约束

    for i in range(bidder):
            model.add_constraint(model.sum([flow[i][j] for j in range(es)])==bw[i]*allocation[i])

    for i in range(bidder):
        for j in range(es):
            model.add_constraint(flow[i][j]<=bwr[i][j])

    for i in range(es):
        model.add_constraint(model.sum([flow[j][i] for j in range(bidder)])<=thp[i])

    solution=model.solve()
    r_payment=0
    for i in range(bidder):
        r_payment+=solution[allocation[i]]*payment[i]
    # print("payment",payment)

    # print(solution)
    if(solution):
        return (solution.objective_value,r_payment)
    else:
        return (-1,-1)

def opsm(bid,value,budget):
    total_value=0
    sorted_indices = np.argsort(-value/bid)
    st=np.zeros_like(value)
    flag=-1
    for i in sorted_indices:
        if bid[i] <= budget / 2 * value[i] / (total_value + value[i]):
            total_value += value[i]
            st[i]=1
            flag=i
        else:
            break

    payment = np.zeros_like(value)
    for i in sorted_indices:
        total_value=0
        if st[i]==0:
            continue
        for j in sorted_indices:
            if i==j :
                continue
            payment[i]=max(payment[i],min(value[i]/value[j]*bid[j],budget/2))
            total_value+=value[j]
            if bid[j] <=value[j]/total_value*budget/2:
                break
        if i==flag:
            payment[i]=bid[i]
    return (value*st,payment*st)

es=3
for bidder in range(2,8):
    for budget in [k*0.5 for k in range(1,8)]:
        path = os.path.join(os.path.join("data", str(bidder) + "x" + str(es)), "train")

        bid = np.load(os.path.join(path, 'bid.npy')).astype(np.float32)
        value = np.load(os.path.join(path,'value.npy')).astype(np.float32)
        bw  = np.load(os.path.join(path, 'bw.npy')).astype(np.float32)
        bwr = np.load(os.path.join(path, 'bwr.npy')).astype(np.float32)
        thp = np.load(os.path.join(path, 'thp.npy')).astype(np.float32)

        bs = bid.shape[0]
        total_utility = 0
        total_payment=0
        for i in tqdm(range(bs)):
            value_,payment=opsm(bid[i],value[i],budget)
            utility,payment=opt_flow(value_,payment,bw[i],bwr[i],thp[i])
            total_utility+=utility-payment
            total_payment+=payment
        logging.info(f"budget={budget},bidder={bidder},es={es},opsm-utility={total_utility/bs},opsm-payment={total_payment/bs},opsm-wf={(total_payment+total_utility)/bs}")
    print()