import os

def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
