import matplotlib.pyplot as plt
mnist_epoch = []
mnist_loss = []

orl_epoch = []
orl_loss = [12.3416,
            484.3569,
            244.3770,
            52.5421,
            5.5754,
            3.2360,
            2.9285,
            2.5634,
            1.9144
            1.3907,
            0.7737,
            0.2507,
            0.1275,
            0.0603,
            0.0342,
            0.0106,
            0.0058,
            0.0014,
            0.0005,
            0.0003,
            0.0002,
            0.0001,
            0.0001,
            0.0001,
            0.0000,
            0.0000]

cifar_epoch = [i for i in range(1, 101)]
cifar_loss = []

def plot_graph(epoch_arr, loss_arr):



