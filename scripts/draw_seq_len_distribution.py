import os
import numpy as np
import matplotlib.pyplot as plt


def load_and_draw(file_path):
    sentence_lengths = np.load(file_path)
    sentence_lengths = np.sort(sentence_lengths)
    length = len(sentence_lengths)
    fig_path = os.path.join(os.path.splitext(os.path.basename(file_path))[0]) 
    plt.hist(sentence_lengths[:int(length * 0.99)], bins=100, cumulative=True, edgecolor='black', alpha=0.7, density=True, histtype='stepfilled')
    plt.title('Cumulative Distribution of Sentence Lengths')
    plt.xlabel('Sentence Length')
    plt.ylabel('Cumulative Frequency')
    plt.grid(True)
    ratio = 0.90
    plt.axhline(y=ratio, color='red', linestyle='--', label=f'{ratio*100}% Line')
    plt.axvline(x=sentence_lengths[int(length * ratio)], color='red', linestyle='--', label=f'{ratio * 100}% Line')
    plt.savefig(fig_path)
    plt.clf()
    
    # ratio = 0.90
    # print(f"{ratio * 100}% of the data has length <= {sentence_lengths[int(length * ratio)]}")
    # ratio = 0.90
    # print(f"{ratio * 100}% of the data has length <= {sentence_lengths[int(length * ratio)]}")


# plt.hist(sentence_lengths_dataset1, bins=20, cumulative=True, edgecolor='black', alpha=0.7, density=True, histtype='stepfilled', label='Dataset 1')
# plt.hist(sentence_lengths_dataset2, bins=20, cumulative=True, edgecolor='black', alpha=0.7, density=True, histtype='stepfilled', label='Dataset 2')

if __name__ == "__main__":
    path = "/workspace/Dataset/decompilation-dataset/AnghaBench-assembly-g-O0-bin/AnghaBench-assembly-g-O0_seq_len.npy"
    load_and_draw(path)
    path = "/workspace/Dataset/decompilation-dataset/AnghaBench-assembly-g-O2-bin/AnghaBench-assembly-g-O2_seq_len.npy"
    load_and_draw(path)
    