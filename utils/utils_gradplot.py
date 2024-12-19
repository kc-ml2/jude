# import matplotlib.pyplot as plt


def plot_grad_flow(named_parameters, save_path):
    ave_grads = []
    layers = []
    f = open("{}/grad.txt".format(save_path), "w")
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            # print(n, p.grad.cpu().abs().mean())
            # if p.grad.cpu().abs().mean() == 0.0:
            #     print(n, p.grad.cpu().abs().mean())
            f.writelines(n + " --- " + str(p.grad.cpu().abs().mean().numpy()) + "\n")
    f.close()
    # plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    # plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    # plt.xlim(xmin=0, xmax=len(ave_grads))
    # plt.xlabel("Layers")
    # plt.ylabel("average gradient")
    # plt.title("Gradient flow")
    # plt.grid(True)
    # plt.savefig("{}/grad.png".format(opt["path"]['save']))
    # plt.close()
