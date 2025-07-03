import logging
from pathlib import Path

import torch
from agent import EncodingAgent
from env import EncodingCheckEnv
from model import Decoder, Encoder
from trainer import VAETrainer

from pamiq_core import FixedIntervalInteraction, LaunchConfig, launch
from pamiq_core.data.impls import RandomReplacementBuffer
from pamiq_core.torch import TorchTrainingModel


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    feature_size = 64

    interaction = FixedIntervalInteraction.with_sleep_adjustor(
        agent=EncodingAgent(),
        environment=EncodingCheckEnv(feature_size=feature_size),
        interval=0.1,  # 10fps
    )

    models = {
        "encoder": TorchTrainingModel(
            Encoder(feature_size=feature_size),
            inference_procedure=Encoder.infer,
            device=device,
        ),
        "decoder": TorchTrainingModel(
            Decoder(feature_size=feature_size),
            has_inference_model=False,
            device=device,
        ),
    }

    buffers = {"observation": RandomReplacementBuffer(max_size=1024)}
    trainers = {"vae": VAETrainer(max_epochs=3, batch_size=32)}

    launch(
        interaction=interaction,
        models=models,
        buffers=buffers,
        trainers=trainers,
        config=LaunchConfig(
            states_dir=Path(__file__).parent / "states",
            web_api_address=("localhost", 8391),  # for `pamiq-console` connection
            max_uptime=300.0,  # 5 minutes
        ),
    )


if __name__ == "__main__":
    main()
