import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--file_path', default='../dataset/train/', type=str, help='main path of dataset')
    parser.add_argument('--train_num', default=13, type=int, help='the number of train pic in a set')
    parser.add_argument('--data_num', default=19, type=int, help='the number of pic in a set')
    parser.add_argument('--epochs', default=100, type=int, help='the number of batch')
    parser.add_argument('--compress_resize', default=16, type=int, help='compressed image length')
    parser.add_argument('--input_size', default=256, type=int, help='the input size to rl')
    parser.add_argument('--output_size', default=256, type=int, help='the output size from rl')
    parser.add_argument('--sim_duration', default=4000, type=float, help='the total simulation time')
    parser.add_argument('--frame_duration', default=200, type=float, help='input encoding time for each image')
    parser.add_argument('--dt', default=1, type=int, help='dt(ms)')
    parser.add_argument('--firing_amplify', default=2, type=int, help='pulse rate amplification factor')
    parser.add_argument('--input_amplify', default=5, type=int, help='the number of neurons in the rl connected by an il neuron')
    parser.add_argument('--tau', default=100, type=float, help='tau')
    parser.add_argument('--window_size', default=200, type=int, help='the size of spike counting window')
    return parser.parse_args()


args = parse_args()