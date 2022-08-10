from pathlib import Path
from easydict import EasyDict
import argparse
import yaml


cfg = EasyDict()
cfg.LOCAL_RANK = 0

def parser_config():
    parser = argparse.ArgumentParser(description='Arg parser')
    # basic parameter for training
    parser.add_argument('--cfg', type=str, default=None, help='specify the config for training')
    parser.add_argument('--source', '-s', type=str, help='source dataset', default='scannet')
    parser.add_argument('--batchsize', '-b', type=int, help='batch size', default=64)
    parser.add_argument('--epochs', '-e', type=int, help='training epoch', default=200)
    parser.add_argument('--gpu', '-g', type=str, help='cuda id', default='0')

    # basic paramter for cpkt-related
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--ckpt_save_interval', type=int, default=5, help='number of training epochs')
    parser.add_argument('--max_ckpt_save_num', type=int, default=50, help='max number of saved checkpoint')
    args = parser.parse_args()
    
    cfg_from_yaml_file(args.cfg, cfg)
    cfg.TAG = Path(args.cfg).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    return args, cfg

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))
        

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.safe_load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=new_config)

    return config
