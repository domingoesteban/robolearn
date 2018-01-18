import os
import argparse
import yaml
import random
import numpy as np


def main():
    # ##################### #
    # Commandline Arguments #
    # ##################### #
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='test1')
    args = parser.parse_args()
    print('command_line args:', args)

    # ############### #
    # Log directories #
    # ############### #
    log_dir = args.log_dir+'/run_'+str(args.run_num)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        os.makedirs(log_dir+'/dir1')
        os.makedirs(log_dir+'/dir2')
    else:
        print("Log directory '%s' already exists!. Using it." % log_dir)

    # ############# #
    # Load Scenario #
    # ############# #
    param_dict = dict()
    param_dict['scenario'] = args.scenario
    param_dict['seed'] = args.seed
    param_dict['run_num'] = args.run_num
    param_dict['log_dir'] = log_dir

    scenario_module = __import__(args.scenario)
    scenario = scenario_module.Scenario(param_dict)

    # ############# #
    # Set variables #
    # ############# #
    random.seed(args.seed)
    np.random.seed(args.seed)
    # TODO: Set also env
    scenario.env.seed(args.seed)

    # ####################### #
    # Dump Parameters to file #
    # ####################### #
    param_dict['task_params'] = scenario.task_params

    with open(log_dir+'/parameters.yaml', 'w') as outfile:
        yaml.dump(param_dict, outfile, default_flow_style=False)


    # scenario.env.reset(condition=0)
    # scenario.env.render(mode='human')

    # ########### #
    # SCRIPT MODE #
    # ########### #
    if args.mode == 'train':
        successful = scenario.train()
    elif args.mode == 'test':
        successful = scenario.test()
    else:
        raise ValueError('Wrong script option')

    if successful:
        print('The script has finished successfully!!!')
    else:
        print('The script has NOT finished successfully!!!')


if __name__ == '__main__':
    main()
