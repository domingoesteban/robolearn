import os


def get_list_from_ordered_dirs(path, prefix):
    string_list = remove_prefix_from_list(get_dirs_with_prefix(path, prefix),
                                          prefix)
    return [int(element) for element in string_list]


def remove_prefix_from_list(input_list, prefix):
    new_list = list(input_list)
    for ii in range(len(new_list)):
        if new_list[ii].startswith(prefix):
            new_list[ii] = new_list[ii][len(prefix):]
    return new_list


def get_dirs_with_prefix(path, prefix):
    all_dirs = sorted([filename for filename in os.listdir(path) if
                filename.startswith(prefix)])
    return all_dirs