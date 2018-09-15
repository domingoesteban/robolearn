import os
import argparse
import yaml
import random
import numpy as np
import shutil
from scenario import Scenario
from builtins import input


def main():
    # ##################### #
    # Commandline Arguments #
    # ##################### #
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='mdgps')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--run_num', type=int, default=0)
    parser.add_argument('--itr', type=int, default=-1)
    parser.add_argument('--cond', type=int, default=0)
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--local', action="store_true")
    parser.add_argument('--log_dir', type=str, default='NOTHING')

    args = parser.parse_args()
    print('command_line args:', args)


    # ############### #
    # Log directories #
    # ############### #
    if args.log_dir == 'NOTHING':
        log_prefix = args.scenario
    else:
        log_prefix = args.log_dir
    log_dir = str(log_prefix)+('_log/run_%02d' % args.run_num)
    if args.mode == 'train':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            # replace_dir = input("Log directory '%s' already exists!. "
            #                     "Press [y/Y] to replace it and continue with "
            #                     "the script"
            #                     "replace it? [y/Y]: " % log_dir)
            replace_dir = 'y'
            # replace_dir = 'y'
            if replace_dir.lower() == 'y':
                shutil.rmtree(log_dir)
                os.makedirs(log_dir)
            else:
                print('Finishing the script!!!')
                exit()
    elif args.mode == 'test':
        if not os.path.exists(log_dir):
            raise ValueError("It does not exist log directory '%s'"
                             % log_dir)
        else:
            print("Testing script with log directory '%s'!" % log_dir)
    elif args.mode == 'eval':
        if not os.path.exists(log_dir):
            raise ValueError("It does not exist log directory '%s'"
                             % log_dir)
        else:
            print("Testing script with log directory '%s'!" % log_dir)
    else:
        raise ValueError('Wrong script option')

    # ############# #
    # Load Scenario #
    # ############# #
    hyperparam_dict = dict()
    hyperparam_dict['scenario'] = args.scenario
    hyperparam_dict['seed'] = args.seed
    hyperparam_dict['run_num'] = args.run_num
    hyperparam_dict['render'] = args.render
    hyperparam_dict['log_dir'] = log_dir

    scenario = Scenario(hyperparam_dict)

    # ############# #
    # Set variables #
    # ############# #
    random.seed(args.seed)
    np.random.seed(args.seed)
    scenario.env.seed(args.seed)


    # ####################### #
    # Dump Parameters to file #
    # ####################### #
    hyperparam_dict['task_params'] = scenario.task_params

    with open(log_dir+'/hyperparameters.yaml', 'w') as outfile:
        yaml.dump(hyperparam_dict, outfile, default_flow_style=False)


    # scenario.env.reset(condition=0)
    # scenario.env.render(mode='human')

    # ########### #
    # SCRIPT MODE #
    # ########### #
    if args.mode == 'train':
        successful = scenario.train()
    elif args.mode == 'test':
        if args.local:
            pol_type = 'local'
        else:
            pol_type = 'global'
        successful = scenario.test_policy(iteration=args.itr,
                                          condition=args.cond,
                                          pol_type=pol_type)
        input("Press a key to close the script")
    elif args.mode == 'eval':
        successful = scenario.eval_dualism()
    else:
        raise ValueError('Wrong script option')

    if successful:
        print('#'*40)
        print('The script has finished successfully!!!')
        print('#'*40)
    else:
        print('The script has NOT finished successfully!!!')


if __name__ == '__main__':
    main()
