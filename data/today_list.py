import pickle
import os
import glob


truth_dir = '../sample_data/truth/'
data_dir = '../sample_data/data/'
def from_tif_fn_list(fns):

    truth_file_list = []
    data_file_list = []
    for fn in fns:
        truth_file_loc = glob.glob(truth_dir +'2023/*/'+fn)
        data_file_loc = glob.glob(data_dir  +'2023/*/'+fn)

        truth_file_list.extend(truth_file_loc)
        data_file_list.extend(data_file_loc)
    data_dict = {'test': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict
tif_fns = ['G16_s20232680016174_e20232680018547_11.tif', 'G16_s20232680016174_e20232680018547_10.tif', 'G16_s20232680016174_e20232680018547_9.tif', 'G16_s20232680016174_e20232680018547_22.tif', 'G16_s20232672326174_e20232672328547_7.tif', 'G16_s20232680016174_e20232680018547_4.tif']

data_dict = from_tif_fn_list(tif_fns)


#print('number of test samples:', len(test_truth_file_list))


with open('today_data_dict.pkl', 'wb') as handle:
    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
