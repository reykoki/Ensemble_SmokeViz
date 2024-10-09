import pickle
import os
import glob

def data_dict_from_tif_filelist(fns, data_dir='./data/'):
    truth_file_list = []
    data_file_list = []
    for fn in fns:
        truth_file_loc = glob.glob(data_dir +'truth/*/*/'+fn)
        data_file_loc = glob.glob(data_dir  +'data/*/*/'+fn)

        truth_file_list.extend(truth_file_loc)
        data_file_list.extend(data_file_loc)
    data_dict = {'test': {'truth': truth_file_list, 'data': data_file_list}}
    return data_dict

#with open('today_data_dict.pkl', 'wb') as handle:
#    pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
