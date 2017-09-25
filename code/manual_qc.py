import argparse
import numpy as np
def manual_qc(num_volumes):
    '''Allows you to manually input indices for corrupted volumes.'''
    y_human = np.zeros(num_volumes)
    for i in range(num_volumes):
        a = input('Vol '+str(i)+'. Enter 0 for OK, 1 for borderline, 2 for bad: \n')
        if a == '1' or a =='2':
            y_human[int(i)] = int(a)
        else:
            y_human[int(i)] = 0
    return y_human



if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_vols',help='Num volumes to QC',type=int)
    args = parser.parse_args()
    y_human = manual_qc(args.num_vols)
    print(y_human)
    #y_human = manual_qc()
    #y_human = np.load('y_manual_103414.npy')    