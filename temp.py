import torch

if __name__ == "__main__":
    a = torch.rand((3, 1))
    b = torch.rand(3).long().view((-1, 1))

    loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
    ll = loss(a, b)

    print(ll.shape)