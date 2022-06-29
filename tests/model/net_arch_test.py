import os
import tempfile

import torch
from hydra.experimental import compose, initialize

from model import make_model
TEST_DIR = tempfile.mkdtemp(prefix="project_tests")


def test_net_arch():
    os.makedirs(TEST_DIR, exist_ok=True)
    with initialize(config_path="../../config/job/train"):
        cfg = compose(config_name="dncnn", overrides=[f"working_dir={TEST_DIR}"])

    net = make_model(cfg.network)

    # TODO: This is example code. You should change this part as you need. You can code this part as forward
    x = torch.rand(8, 3, 64, 64)
    out = net(x)
    assert out.shape == (8, 3, 64, 64)
    # x = net.conv2(x)  # x: (B,4,7,7)
    # assert x.shape == (8, 4, 7, 7)
    # x = torch.flatten(x, 1)  # x: (B,4*7*7)
    # assert x.shape == (8, 4 * 7 * 7)
    # x = net.fc(x)  # x: (B,10)
    # assert x.shape == (8, 10)
