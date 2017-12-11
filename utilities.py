import os
import torch


def save_checkpoint(state, _dir, max_model=5):
    filename = 'checkpoint_' + str(state['step']) + '.pth.tar'
    txtname = 'checkpoint.txt'
    file_dir = os.path.join(_dir, filename)
    txt_dir = os.path.join(_dir, txtname)

    torch.save(state, file_dir)  # Save checkpoint

    # Append checkpoint name in txt
    with open(txt_dir, 'a') as f_txt:
        f_txt.write(filename + '\n')
    # Delete checkpoint if the number of file exceeded
    with open(txt_dir, 'r') as f_txt:
        f_lines = f_txt.readlines()
        tmp_lines = [a.strip('\n') for a in f_lines]
        if len(f_lines) > max_model:
            tmp_name = tmp_lines.pop(0)
            f_lines.pop(0)
            if os.path.isfile(os.path.join(_dir, tmp_name)):
                os.remove(os.path.join(_dir, tmp_name))
    # Rewrite file
    with open(txt_dir, 'w') as f_txt:
        write_names = ''.join(f_lines)
        f_txt.write(write_names)


def read_checkpoint(_dir):
    txtname = 'checkpoint.txt'
    txt_dir = os.path.join(_dir, txtname)
    if not os.path.isfile(txt_dir):
        checkpoint = None
        is_exist = False
        return is_exist, checkpoint
    print txt_dir
    with open(txt_dir, 'r') as f_txt:
        f_lines = f_txt.readlines()
        tmp_lines = [a.strip('\n') for a in f_lines]
        print tmp_lines
        name = tmp_lines[-1]

    checkpoint_file = os.path.join(_dir, name)
    if os.path.isfile(checkpoint_file):
        print('Reading checkpoint file: ' + name)
        checkpoint = torch.load(checkpoint_file)
        is_exist = True
    else:
        print ('Checkpoint file ' + name + ' not exist!')
        checkpoint = None
        is_exist = False
    return is_exist, checkpoint
