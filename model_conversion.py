from torch import nn
import torch.onnx
import onnx
from onnx_tf.backend import prepare

from pathlib import Path

from networks.network_loader import load_network
from experiment_manager.config.config import load_cfg


def create_model_input(cfg) -> torch.tensor:
    # batch size is later set to dynamic
    batch_size = 1
    n_bands = len(cfg.DATALOADER.SENTINEL1_BANDS) + len(cfg.DATALOADER.SENTINEL2_BANDS)
    model_input = torch.randn(batch_size, n_bands, 256, 256, requires_grad=True)
    return model_input

# see https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
def save_torch_as_onnx(cfg, torch_net, onnx_file: Path, model_check=False):

    x = create_model_input(cfg)
    if model_check:
        torch_net(x)

    # Export the model
    torch.onnx.export(
        torch_net,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        onnx_file,  # where to save the model (can be a file or file-like object)
        verbose=True,
        export_params=True,  # store the trained parameter weights  inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},  # variable lenght axes
            'output': {0: 'batch_size'}
        }
    )


# see https://quq99.github.io/blog/2019-08/onnx-convert-trained-pytorch-model-to-tensorflow-model/
def save_onnx_as_tf(onnx_net, tf_file: Path):
    # load the onnx file

    # Check the model
    onnx.checker.check_model(onnx_net)
    # print('The model before conversion:\n{}'.format(onnx_or_model))

    # # A full list of supported adapters can be found here:
    # # https://github.com/onnx/onnx/blob/master/onnx/version_converter.py#L21
    # # Apply the version conversion on the original model
    # onnx_model = version_converter.convert_version(onnx_or_model, 7)

    # print('The model after conversion:\n{}'.format(onnx_model))

    # import onnx to TF model
    tf_rep = prepare(onnx_net)
    tf_rep.export_graph(tf_file)


def torch2tf(cfg_name: str, cfg_file=None, net_file=None):

    # loading config
    if cfg_file is None:
        cfg_file = Path.cwd() / 'configs' / f'{cfg_name}.yaml'
    cfg = load_cfg(cfg_file)

    # loading network
    if net_file is None:
        net_dir = Path('/storage/shafner/run_logs/unet/')
        net_file = net_dir / cfg_name / 'best_net.pkl'
    net = load_network(cfg, net_file)

    # convert to onnx and save file
    onnx_file = net_file.parent / f'{net_file.stem}.onnx'
    save_torch_as_onnx(cfg, net, onnx_file)

    # load onnx model, convert to tensorflow and save
    onnx_net = onnx.load(onnx_file)
    tf_file = net_file.parent / f'{net_file.stem}.pb'
    save_onnx_as_tf(onnx_net, tf_file)





if __name__ == '__main__':

    torch2tf('baseline_sentinel2')
