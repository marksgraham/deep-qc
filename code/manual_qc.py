import argparse
import numpy as np


def manual_qc(num_volumes):
    '''Allows you to manually input indices for corrupted volumes.'''
    y_human = np.zeros(num_volumes)
    for i in range(num_volumes):
        a = input('Vol '+str(i)+'. Enter 0 for OK, 1 for borderline, 2 for bad, 3 for other or ''e'' to edit previous: \n')
        #Edit mode
        if a =='e':
            index_to_edit=0
            while(index_to_edit!='q'):
                index_to_edit = input('Edit mode. Enter index to alter or ''q'' to quit: \n')
                if index_to_edit == 'q':
                    a = input('Vol '+str(i)+'. Enter 0 for OK, 1 for borderline, 2 for bad, 3 for other: \n')
                    break
                else:
                    print('Vol',index_to_edit,'has value',y_human[int(index_to_edit)])
                    new_value = input('Enter new value: \n')
                    y_human[int(index_to_edit)] = int(new_value)

        if a == '1' or a =='2' or a =='3':
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
    np.save('y_manual.npy',y_human)
    #y_human_loaded = np.load('y_manual.npy')    
    #print(y_human_loaded)