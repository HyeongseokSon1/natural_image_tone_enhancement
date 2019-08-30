from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 9
config.TRAIN.lr_init = 1e-4
config.TRAIN.lr_decay = 0.1
config.TRAIN.beta1 = 0.9

# various log location
config.TRAIN.checkpoint = 'checkpoint/'
config.TRAIN.save_valid_results = 'result_valid' 
config.TRAIN.summary_g = 'summary_init'
config.TRAIN.summary_adv = 'summary'

## train set location
config.TRAIN.hr_img_path = '/home/sonhs/data1/5K_small/set2/result2/' #need to change
config.TRAIN.lr_img_path = '/home/sonhs/data1/5K_small/set2/image/' #need to change
config.TRAIN.real_img_path = '/home/sonhs/data1/5K_small/set2/hdr_resize2/' #need to change

config.VALID = edict()
## validation set location
config.VALID.hr_img_path = '/home/sonhs/data1/SRdataset/valid/GT/' #need to change
config.VALID.lr_img_path = '/home/sonhs/data1/SRdataset/valid/LR_bicubic/' #need to change

config.TEST = edict()
config.TEST.checkpoint = 'models/'
config.TEST.input_path = 'real/' #need to change
config.TEST.save_path = 'real_eval/'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
