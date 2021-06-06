import pandas as pd
import glob
import argparse

parser = argparse.ArgumentParser(description='preprocessing')

parser.add_argument('--split', help='train/test/val', type=str)
parser.add_argument('--num', help='batch number', type=int)

args = parser.parse_args()

SPLIT = args.split
NUM = args.num

if __name__ == '__main__':

    names = glob.glob(f"./{SPLIT}/{NUM}/*.jpg")
    names = [int(n.replace(f'./{SPLIT}/{NUM}/', '').replace('.jpg', '')) for n in names]

    df = pd.read_csv(f'./{SPLIT}/{NUM}.csv', index_col=0)
    df_after = df.loc[df['number'].isin(names)]
    df_after.to_csv(f'./{SPLIT}/{NUM}_after.csv', index=False)