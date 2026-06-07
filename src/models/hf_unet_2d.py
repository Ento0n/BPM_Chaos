from diffusers import UNet2DModel


def create_model(sample_size: int = 256) -> UNet2DModel:
    # Build a grayscale UNet without attention blocks to keep Mac memory use lower.
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


model = create_model()
