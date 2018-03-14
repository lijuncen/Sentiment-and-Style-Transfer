from collections import OrderedDict
import cPickle
import os
import hashlib



def save_params_val(path, tparams):
    """
    Save the instance params into disk.
    NOTICE that the data are compressed by cPickle.
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    param_dict = OrderedDict()
    for pname in tparams :
        param_dict[pname] = tparams[pname].get_value()
    with open(path, 'wb') as fw:
        cPickle.dump(param_dict, fw, protocol=cPickle.HIGHEST_PROTOCOL)  # serialize and save object


def load_params_val(path):
    """
    Load the instance params from disk.
    """

    param_dict = OrderedDict()
    if(not os.path.exists(path)):
        return None
    with open(path, 'rb') as f:  # open file with write-mode
        while f:
            try:
                param_dict = cPickle.load(f)
            except:
                break
    return param_dict


def get_params_file_name(conf_dict) :
    """
    Get the name of params file name, by hashing the conf_dict.
    """
    conf_str = ','.join(['{0}:{1}'.format(key, value) for key, value in conf_dict.iteritems()])
    return conf_dict['algo_name'] + '_' + hashlib.md5(conf_str).hexdigest()


def save_confs_val(conf_dict, path):
    """
    Save the configuration of the model into disk.
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    charset = conf_dict['charset']
    with open(path, 'wb') as fw:
        for conf in conf_dict:
            fw.writelines((str(conf) + ' = ' + str(conf_dict[conf]) + '\n').encode(charset))