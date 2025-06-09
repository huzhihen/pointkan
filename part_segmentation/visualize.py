"""
Plot the parts.
python visualize.py --model PointKAN --exp_name demo1
"""
from __future__ import print_function
import argparse
import torch
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from util.data_util import PartNormalDataset
import model as models
import numpy as np
from util.util import to_categorical
import os


colrs_list = [
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "deepskyblue", "m", "deeppink", "hotpink", "lime", "c", "y",
    "gold", "darkorange", "g", "orangered", "tomato", "tan", "darkorchid", "violet",
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "deepskyblue", "m", "deeppink", "hotpink", "lime", "c", "y",
    "gold", "darkorange", "g", "orangered", "tomato", "tan", "darkorchid", "violet",
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "deepskyblue", "m", "deeppink", "hotpink", "lime", "c", "y",
    "gold", "darkorange", "g", "orangered", "tomato", "tan", "darkorchid", "violet"
]


def test(args):
    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("===> The number of test data is:%d", len(test_data))

    # Try to load models
    print("===> Create model...")
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")
    model = models.__dict__[args.model](num_part).to(device)

    # Load checkpoint
    print("===> Load checkpoint...")
    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']
    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    # Start evaluate
    print("===> Start evaluate...")
    model.eval()
    num_classes = 16
    save_dir = os.path.join("figures", args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(test_data)):
        points, label, target, norm_plt = test_data.__getitem__(i)
        points = torch.tensor(points).unsqueeze(dim=0)
        label = torch.tensor(label).unsqueeze(dim=0)
        target = torch.tensor(target).unsqueeze(dim=0)
        norm_plt = torch.tensor(norm_plt).unsqueeze(dim=0)
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.to(device), label.squeeze(dim=0).to(device), target.to(device), norm_plt.to(device)

        with torch.no_grad():
            cls_lable = to_categorical(label, num_classes)
            predict = model(points, norm_plt, cls_lable)  # b,n,50

        # up to now, points [1, 3, 2048]  predict [1, 2048, 50] target [1, 2048]
        predict = predict.max(dim=-1)[1]
        predict = predict.squeeze(dim=0).cpu().data.numpy()  # 2048
        target = target.squeeze(dim=0).cpu().data.numpy()  # 2048
        points = points.transpose(2, 1).squeeze(dim=0).cpu().data.numpy()  # [2048,3]

        # save point.txt file
        np.savetxt(os.path.join(save_dir, f"{i}-point.txt"), points)
        np.savetxt(os.path.join(save_dir, f"{i}-target.txt"), target)
        np.savetxt(os.path.join(save_dir, f"{i}-predict.txt"), predict)

        # start plot
        print(f"===> stat plotting sample {i}")
        plot_xyz(points, target, name=os.path.join(save_dir, f"{i}-gt.pdf"))
        plot_xyz(points, predict, name=os.path.join(save_dir, f"{i}-predict.pdf"))


def plot_xyz(xyz, target, name="figures/figure.pdf"):
    fig = pyplot.figure()
    ax = Axes3D(fig)
    # ax = fig.gca(projection='3d')
    x_vals = xyz[:, 0]
    y_vals = xyz[:, 1]
    z_vals = xyz[:, 2]
    ax.set_xlim3d(min(x_vals)*0.9, max(x_vals)*0.9)
    ax.set_ylim3d(min(y_vals)*0.9, max(y_vals)*0.9)
    ax.set_zlim3d(min(z_vals)*0.9, max(z_vals)*0.9)

    ax.view_init(elev=0, azim=-90)

    for i in range(0, 2048):
        col = int(target[i])
        ax.scatter(x_vals[i], y_vals[i], z_vals[i], c=colrs_list[col], marker="o", s=30, alpha=0.7)
    ax.set_axis_off()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    # pyplot.tight_layout()
    fig.savefig(name, bbox_inches='tight', pad_inches=-0.3, transparent=True)
    pyplot.close()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='PointKAN')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model + "_" + args.exp_name
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    test(args)