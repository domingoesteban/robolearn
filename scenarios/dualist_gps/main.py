import os
import argparse
import yaml
import random
import numpy as np
import shutil


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
    # log_dir = args.log_dir
    log_dir = args.log_dir+('/run_%02d' % args.run_num)
    if args.mode == 'train':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            replace_dir = input("Log directory '%s' already exists!. "
                                "Press [y/Y] to replace it and continue with "
                                "the script"
                                "replace it? [y/Y]: " % log_dir)
            # replace_dir = 'y'
            if replace_dir.lower() == 'y':
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
            else:
                print('Finishing the script!!!')
                exit()

    elif args.mode == 'test':
        print("Testing script with log directory '%s'!")
    else:
        raise ValueError('Wrong script option')

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
