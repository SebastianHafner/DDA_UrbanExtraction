from networks.network_loader import create_network
from experiment_manager.args import default_argument_parser
from experiment_manager.config import config


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':

    args = default_argument_parser().parse_known_args()[0]
    cfg = config.setup(args)
    net = create_network(cfg)

    n_parameters = count_parameters(net)
    print(n_parameters)


