import torch
import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys

def get_df(outdir):
    #Gather all pth files
    files = glob.glob(f"{outdir}\\save_*.pth")

    data = []
    for file in files:
        d = torch.load(file)
        del(d['net'])
        data.append(d)

    df = pd.DataFrame(data)
    df.set_index('epoch', inplace=True)
    df.sort_index(inplace=True)
    return df

def make_plot(outdir, metric):
    df = get_df(outdir)
    plt.figure()
    df[f"train_{metric}"].plot()
    df[f'val_{metric}'].plot()
    plt.legend(loc='best')
    plt.savefig(f"{outdir}\\{metric}.png")

def main():
    if len(sys.argv) == 2:
        for metric in ['loss', 'accuracy', 'recall', 'balanced_accuracy', 'precision', 'F1-score']:
            make_plot(outdir=sys.argv[1], metric=metric)
    else:
        make_plot(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()

