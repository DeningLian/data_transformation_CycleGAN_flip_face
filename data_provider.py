from nori.dp import make_dp
from IPython import embed

def create_data_loader(dataset):
    print(dataset)
    data_loader = make_dp(dataset)
    return data_loader


