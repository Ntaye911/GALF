import os
import numpy as np
import urllib.request
from torch_geometric.datasets import Planetoid, PPI
from sklearn.neighbors import kneighbors_graph
from scipy.io import arff

def download_file(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {dest}...")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            with open(dest, 'wb') as f:
                f.write(response.read())

def load_cora_ml():
    url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/cora_ml.npz"
    path = "data/cora_ml.npz"
    download_file(url, path)
    loader = np.load(path, allow_pickle=True)
    X = loader['attr_data'].astype(np.float32)
    A = loader['adj_data'].astype(np.float32)
    labels = loader['labels'].astype(np.int64)
    labels = np.eye(labels.max() + 1)[labels]
    return X, A, labels

def load_pubmed_mc():
    dataset = Planetoid(root='data/PubMed', name='PubMed')
    data = dataset[0]
    X = data.x.numpy().astype(np.float32)
    edge_index = data.edge_index.numpy()
    n = X.shape[0]
    A = np.zeros((n, n))
    A[edge_index[0], edge_index[1]] = 1
    A = ((A + A.T) > 0).astype(np.float32)
    labels = np.eye(dataset.num_classes)[data.y.numpy()]
    return X, A, labels

def load_ppi():
    train_dataset = PPI(root='data/PPI', split='train')
    val_dataset = PPI(root='data/PPI', split='val')
    test_dataset = PPI(root='data/PPI', split='test')
    data_list = list(train_dataset) + list(val_dataset) + list(test_dataset)
    X = np.vstack([d.x.numpy() for d in data_list]).astype(np.float32)
    Y = np.vstack([d.y.numpy() for d in data_list]).astype(np.float32)
    edge_index = np.hstack([d.edge_index.numpy() for d in data_list])
    n = X.shape[0]
    A = np.zeros((n, n))
    A[edge_index[0], edge_index[1]] = 1
    A = ((A + A.T) > 0).astype(np.float32)
    return X, A, Y

def load_mulan_dataset(name):
    base_url = "http://mulan.sourceforge.net/datasets-mllc/"
    arff_file = f"{name}.arff"
    xml_file = f"{name}.xml"
    download_file(base_url + arff_file, f"data/{arff_file}")
    download_file(base_url + xml_file, f"data/{xml_file}")
    
    data, meta = arff.loadarff(f"data/{arff_file}")
    feature_cols = [col for col in meta.names() if not col.startswith('label')]
    label_cols = [col for col in meta.names() if col.startswith('label')]
    
    X = np.array([list(row)[:len(feature_cols)] for row in data], dtype=np.float32)
    Y_str = np.array([list(row)[len(feature_cols):] for row in data])
    
    Y_list = []
    for row in Y_str:
        labels = [lbl.decode() if isinstance(lbl, bytes) else lbl for lbl in row]
        Y_list.append([1 if lbl.strip() == '1' else 0 for lbl in labels])
    Y = np.array(Y_list, dtype=np.float32)
    
    A_sparse = kneighbors_graph(X, n_neighbors=5, mode='connectivity', include_self=False)
    A = A_sparse.todense().astype(np.float32)
    A = ((A + A.T) > 0).astype(np.float32)
    return X, A, Y

def load_yeast(): return load_mulan_dataset("yeast")
def load_delicious(): return load_mulan_dataset("delicious")
def load_eurlex(): return load_mulan_dataset("eurlex")

DATASETS = {
    'Cora-ML': load_cora_ml,
    'PubMed-MC': load_pubmed_mc,
    'PPI': load_ppi,
    'Yeast': load_yeast,
    'Delicious': load_delicious,
    'EurLEX': load_eurlex
}