import numpy as np
import os


def get_data(n_bidder,m_item,path,n_data):

    value=np.random.uniform(0,3,size=(n_data,n_bidder))
    bid=np.random.uniform(0,1,size=(n_data,n_bidder))
    bw=np.random.uniform(0,4,size=(n_data,n_bidder))
    bwr=np.random.uniform(0,1,size=(n_data,n_bidder,m_item))
    thp=np.random.uniform(0,5,size=(n_data,m_item))

    path_dir=os.path.join("data",str(n_bidder)+'x'+str(m_item))
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    path_dir = os.path.join(path_dir, path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    # print(bwr.shape)
    np.save(os.path.join(path_dir,"value"), value, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "bid"), bid, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir,"bw"), bw, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "bwr"), bwr, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(path_dir, "thp"), thp, allow_pickle=True, fix_imports=True)

print("generate data:")

train_dir="train"
test_dir="test"

n=2
m=3
train_n_head=1e5
test_n_head=5000
get_data(int(n),int(m),train_dir,int(train_n_head))
get_data(int(n),int(m),test_dir,int(test_n_head))
print("user={},es={}.".format(n,m))

n=3
m=3
train_n_head=1e5
test_n_head=5000
get_data(int(n),int(m),train_dir,int(train_n_head))
get_data(int(n),int(m),test_dir,int(test_n_head))
print("user={},es={}.".format(n,m))

n=4
m=3
train_n_head=1e5
test_n_head=5000
get_data(int(n),int(m),train_dir,int(train_n_head))
get_data(int(n),int(m),test_dir,int(test_n_head))
print("user={},es={}.".format(n,m))

n=5
m=3
train_n_head=1e5
test_n_head=5000
get_data(int(n),int(m),train_dir,int(train_n_head))
get_data(int(n),int(m),test_dir,int(test_n_head))
print("user={},es={}.".format(n,m))

n=6
m=3
train_n_head=1e5
test_n_head=5000
get_data(int(n),int(m),train_dir,int(train_n_head))
get_data(int(n),int(m),test_dir,int(test_n_head))
print("user={},es={}.".format(n,m))

n=7
m=3
train_n_head=1e5
test_n_head=5000
get_data(int(n),int(m),train_dir,int(train_n_head))
get_data(int(n),int(m),test_dir,int(test_n_head))
print("user={},es={}.".format(n,m))