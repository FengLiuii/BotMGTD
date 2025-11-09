import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
import dgl
from Utils.TimeLogger import log
import torch as t
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
device = "cuda:0" if t.cuda.is_available() else "cpu"

class DataHandler:
    def __init__(self):

        if args.data == 'Twibot22':
            predir = './data/Twibot22/'
        elif args.data == 'MGTAB':
            predir = './data/MGTAB/'
        elif args.data == 'Twibot20':
            predir = './data/Twibot20/'
        else:
            raise ValueError(f"Unknown dataset: {args.data}")
        self.predir = predir  

    def loadOneFile(self, filename):
        with open(filename, 'rb') as fs: 
            ret = (pickle.load(fs) != 0).astype(np.float32)  
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret



    def normalizeAdj(self, mat):
        degree = np.array(mat.sum(axis=-1))  
        dInvSqrt = np.power(degree, -0.5).flatten()
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0  
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        # make ui adj
        user,item = mat.shape[0],mat.shape[1] 
        a = sp.csr_matrix((user, user)) 
        b = sp.csr_matrix((item, item)) 
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])]) 
        mat = (mat != 0) * 1.0 
        # mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat) 

        # make cuda tensor
       
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(device)

    def makeTorchuAdj(self, mat): 
        """Create tensor-based adjacency matrix for user social graph.

        Args:
            mat: Adjacency matrix.

        Returns:
            Tensor-based adjacency matrix.
        """
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0  
        
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).to(device)
    def makeBiAdj(self, mat):
        n_user = mat.shape[0]
        n_item = mat.shape[1]
        a = sp.csr_matrix((n_user, n_user))
        b = sp.csr_matrix((n_item, n_item))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = mat.tocoo()
        edge_src,edge_dst = mat.nonzero()
        ui_graph = dgl.graph(data=(edge_src, edge_dst),
                            idtype=t.int32,
                             num_nodes=mat.shape[0]
                             )

        return ui_graph


    
    def LoadData(self):
        
        if args.data == 'MGTAB':  
            
            (self.feature_list, self.train_idx, self.val_idx, self.test_idx, self.labels, self.he_adjs) = self.load_mgtab_data()
        
        elif args.data == 'Twibot20':  
        
            (self.feature_list, self.train_idx, self.val_idx, self.test_idx, self.labels, self.he_adjs) = self.load_mgtab_data()

        
        elif args.data == 'Twibot22':  
        
            (self.feature_list, self.train_idx, self.val_idx, self.test_idx, self.labels, self.he_adjs) = self.load_mgtab_data()

        self.train_idx = train
        self.val_idx = val
        self.test_idx = test
        self.labels = labels

           


    def load_mgtab_data(self):
       
        self.predir = './data/MGTAB/'  

        
        features_a = sp.load_npz(self.predir + 'a_feat.npz').astype("float32")  
        features_a = t.FloatTensor(preprocess_features(features_a)).to(device)  

      
        follower_mat = sp.load_npz(self.predir + "follower.npz")
        friend_mat = sp.load_npz(self.predir + "friend.npz")
        mention_mat = sp.load_npz(self.predir + "mention.npz")
        reply_mat = sp.load_npz(self.predir + "reply.npz")
        quote_mat = sp.load_npz(self.predir + "quote.npz")
        other_mat = sp.load_npz(self.predir + "other.npz")

       
        labels = np.load(self.predir + 'labels.npy')

       
        if labels.ndim == 1:  
            labels = encode_onehot(labels)

        labels = t.FloatTensor(labels).to(device)  

        
        num_samples = labels.shape[0]  
        indices = np.arange(num_samples)  

        
        if not os.path.exists(self.predir):
            os.makedirs(self.predir)

       
        if not os.path.exists(self.predir + f"train_{int(args.ratio[0]*10)}.npy"):
            print("🔹 No pre-split data found. Performing automatic data partitioning...")
            for ratio in args.ratio:  
                train_idx, temp_idx = train_test_split(
                    indices, test_size=1 - ratio, random_state=42, stratify=labels.cpu().numpy()
                )
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=1/3, random_state=42, stratify=labels.cpu().numpy()[temp_idx]
                )

                np.save(self.predir + f"train_{int(ratio*10)}.npy", train_idx)  # 0.2 → train_2.npy
                np.save(self.predir + f"val_{int(ratio*10)}.npy", val_idx)
                np.save(self.predir + f"test_{int(ratio*10)}.npy", test_idx)

                print(f" data split {ratio} saved | train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")

        
        train_idx = [np.load(self.predir + f"train_{int(i*10)}.npy") for i in args.ratio]
        val_idx = [np.load(self.predir + f"val_{int(i*10)}.npy") for i in args.ratio]
        test_idx = [np.load(self.predir + f"test_{int(i*10)}.npy") for i in args.ratio]

        
        train_idx = [t.LongTensor(i).to(device) for i in train_idx]
        val_idx = [t.LongTensor(i).to(device) for i in val_idx]
        test_idx = [t.LongTensor(i).to(device) for i in test_idx]

        print(f" data loaded successfully | train: {len(train_idx[0])} | val: {len(val_idx[0])} | test: {len(test_idx[0])}")

        
        self.hete_adj1 = follower_mat
        self.hete_adj2 = friend_mat
        self.hete_adj3 = mention_mat
        self.hete_adj4 = reply_mat
        self.hete_adj5 = quote_mat
        self.hete_adj6 = other_mat

        self.he_adjs = [
            dgl.from_scipy(self.hete_adj1).to(device),
            dgl.from_scipy(self.hete_adj2).to(device),
            dgl.from_scipy(self.hete_adj3).to(device),
            dgl.from_scipy(self.hete_adj4).to(device),
            dgl.from_scipy(self.hete_adj5).to(device),
            dgl.from_scipy(self.hete_adj6).to(device)
        ]
        self.feature_list = features_a
        self.train_idx = train_idx 
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.labels = labels
        
        return self.feature_list, self.train_idx, self.val_idx, self.test_idx, self.labels, self.he_adjs



    def load_twibot20_data(self):
        """
            Load the Twibot20 dataset and partition it into training, validation, 
            and test sets with a 7:1:2 ratio:
            - Feature matrix (a_feat.npz)
            - 2 types of adjacency matrices (follower, friend)
            - Train/validation/test splits (7:2:1). If no pre-divided data is found, 
            the dataset will be automatically partitioned and saved.
            - Label data (labels.npy)
        """
        self.predir = './data/Twibot20/' 

        
        features_a = sp.load_npz(self.predir + 'a_feat.npz').astype("float32")  
        features_a = t.FloatTensor(preprocess_features(features_a)).to(device)  

       
        follower_mat = sp.load_npz(self.predir + "follower.npz")
        friend_mat = sp.load_npz(self.predir + "friend.npz")
      

        
        labels = np.load(self.predir + 'labels.npy')

        if labels.ndim == 1:  
            labels = encode_onehot(labels)

        labels = t.FloatTensor(labels).to(device) 

       
        num_samples = labels.shape[0]  
        indices = np.arange(num_samples) 

        
        if not os.path.exists(self.predir):
            os.makedirs(self.predir)

        
        if not os.path.exists(self.predir + f"train_{int(args.ratio[0]*10)}.npy"):
            print("🔹 No pre-split data found. Performing automatic data partitioning...")
            for ratio in args.ratio:  
                train_idx, temp_idx = train_test_split(
                    indices, test_size=1 - ratio, random_state=42, stratify=labels.cpu().numpy()
                )
                val_idx, test_idx = train_test_split(
                    temp_idx, test_size=2/3, random_state=42, stratify=labels.cpu().numpy()[temp_idx]
                )

                
                np.save(self.predir + f"train_{int(ratio*10)}.npy", train_idx)  # 0.2 → train_2.npy
                np.save(self.predir + f"val_{int(ratio*10)}.npy", val_idx)
                np.save(self.predir + f"test_{int(ratio*10)}.npy", test_idx)

                print(f" data split {ratio} saved | train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}")

        
        train_idx = [np.load(self.predir + f"train_{int(i*10)}.npy") for i in args.ratio]
        val_idx = [np.load(self.predir + f"val_{int(i*10)}.npy") for i in args.ratio]
        test_idx = [np.load(self.predir + f"test_{int(i*10)}.npy") for i in args.ratio]

        
        train_idx = [t.LongTensor(i).to(device) for i in train_idx]
        val_idx = [t.LongTensor(i).to(device) for i in val_idx]
        test_idx = [t.LongTensor(i).to(device) for i in test_idx]

        print(f" data loaded successfully | train: {len(train_idx[0])} | val: {len(val_idx[0])} | test: {len(test_idx[0])}")

       
        self.hete_adj1 = follower_mat
        self.hete_adj2 = friend_mat
       

        self.he_adjs = [
            dgl.from_scipy(self.hete_adj1).to(device),
            dgl.from_scipy(self.hete_adj2).to(device),
        ]
        self.feature_list = features_a
        self.train_idx = train_idx 
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.labels = labels
        
        return self.feature_list, self.train_idx, self.val_idx, self.test_idx, self.labels, self.he_adjs

        



def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()
def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot




class index_generator:
    def __init__(self, batch_size, indices=None, shuffle=True):
        self.num_data = len(indices)
        self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(self.indices[(self.iter_counter - 1) * self.batch_size:self.iter_counter * self.batch_size])

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
