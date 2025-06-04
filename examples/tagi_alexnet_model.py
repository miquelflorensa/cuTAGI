from pytagi.nn import (
    AvgPool2d,
    Conv2d,
    Linear,
    MaxPool2d,
    MixtureReLU,
    ReLU,
    Sequential,
    EvenExp,
)


def create_alexnet(
    gain_w: float = 1, gain_b: float = 1, nb_outputs: int = 1001
):
    alex_net = Sequential(
        # 224x224
        Conv2d(
            3,
            64,
            12,
            stride=4,
            padding=2,
            gain_weight=gain_w,
            gain_bias=gain_b,
            in_width=224,
            in_height=224,
            bias=False,
        ),
        ReLU(),
        # 55x55
        AvgPool2d(3, 2),
        # 27x27
        Conv2d(
            64,
            192,
            5,
            bias=False,
            padding=2,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        ReLU(),
        # 27x27
        AvgPool2d(3, 2),
        # 13x13
        Conv2d(
            192,
            384,
            3,
            bias=False,
            padding=1,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        # 13x13
        ReLU(),
        # 13x13
        Conv2d(
            384,
            256,
            3,
            bias=False,
            padding=1,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        ReLU(),
        # 13x13
        Conv2d(
            256,
            256,
            3,
            bias=False,
            padding=1,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        ReLU(),
        # 13x13
        AvgPool2d(3, 2),
        # 6x6
        Linear(256 * 6 * 6, 4096, gain_weight=gain_w, gain_bias=gain_b),
        ReLU(),
        Linear(4096, 4096, gain_weight=gain_w, gain_bias=gain_b),
        ReLU(),
        Linear(4096, nb_outputs, gain_weight=gain_w, gain_bias=gain_b),
        EvenExp(),
    )

    return alex_net

def create_alexnet_cifar100(
    gain_w: float = 1, gain_b: float = 1, nb_outputs: int = 200
):
    alex_net = Sequential(
        # 32x32
        Conv2d(
            3,
            64,
            3,
            stride=1,
            padding=1,
            gain_weight=gain_w,
            gain_bias=gain_b,
            in_width=32,
            in_height=32,
            bias=False,
        ),
        ReLU(),
        # 32x32
        AvgPool2d(2, 2),
        # 16x16
        Conv2d(
            64,
            192,
            3,
            bias=False,
            padding=1,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        ReLU(),
        # 16x16
        AvgPool2d(2, 2),
        # 8x8
        Conv2d(
            192,
            384,
            3,
            bias=False,
            padding=1,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        ReLU(),
        # 8x8
        Conv2d(
            384,
            256,
            3,
            bias=False,
            padding=1,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        ReLU(),
        # 8x8
        Conv2d(
            256,
            256,
            3,
            bias=False,
            padding=1,
            gain_weight=gain_w,
            gain_bias=gain_b,
        ),
        ReLU(),
        # 8x8
        AvgPool2d(2, 2),
        # 4x4
        Linear(256 * 4 * 4, 4096, gain_weight=gain_w, gain_bias=gain_b),
        ReLU(),
        Linear(4096, 4096, gain_weight=gain_w, gain_bias=gain_b),
        ReLU(),
        Linear(4096, nb_outputs, gain_weight=gain_w, gain_bias=gain_b),
        EvenExp(),
    )

    return alex_net
