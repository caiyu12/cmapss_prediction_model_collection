import argparse
def configs():
    parser = argparse.ArgumentParser()

    # general configuration
    # parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument("--sen_len", type=int, default=30)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--enc_in', type=int, default=14)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--pred_len', type=int, default=1)