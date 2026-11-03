import argparse
import os
import logging
from pathlib import Path

import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from facto.encoder import Encoder
from facto.estimator import ContentEstimator, KMeans
from facto.decoder import Decoder
from facto.dataset import AudioDataset
from facto.discriminators import (
    Discriminator,
    hinge_loss,
    hinge_loss_adversarial,
    feature_matching_loss,
)
from facto.melloss import MultiResolutionMelSpectralLoss
from facto.scheduler import InverseLR
from facto.logging import LossMeter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


HIDDEN_SIZE = 1024  # WavLM feature size
SPEAKER_SIZE = 512  # speaker embedding size

# Training schedule (Table 7 of the paper). Pretraining runs on continuous
# content; passing --kmeans switches to fine-tuning on tokens.
PRETRAIN_STEPS = 2_000_000
PRETRAIN_LR = 2e-4
FINETUNE_STEPS = 500_000
FINETUNE_LR = 1e-4

BATCH_SIZE = 16  # per GPU
NUM_WORKERS = 8  # data loading workers per GPU

# Optimizers (AdamW) and inverse-power learning rate schedule.
GEN_BETAS = (0.9, 0.95)
DISC_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-5
ADAM_EPS = 1e-10
LR_DECAY = 50000  # inverse gamma
LR_POWER = 0.5
LR_WARMUP = 0.999  # geometric warmup
CLIP_GRAD_NORM = 10

# Generator loss: adversarial + MATCHING_WEIGHT * feature matching + MEL_WEIGHT * mel.
MATCHING_WEIGHT = 2
MEL_WEIGHT = 45

LOG_INTERVAL = 200  # steps between training log entries

# Decoder upsampling for each patch size (frames per token). The upsampling
# factors multiply to 320 * patch_size, i.e. samples per token at 16 kHz.
DECODER_CONFIGS = {
    4: dict(upsample_factors=(10, 8, 8, 2), upsample_kernel_sizes=(20, 16, 16, 4)),
    2: dict(upsample_factors=(10, 8, 4, 2), upsample_kernel_sizes=(20, 16, 8, 4)),
}


def load_content_estimator(args) -> tuple[ContentEstimator, KMeans | None]:
    """Build the frozen content estimator (and codebook) from fitted checkpoints."""
    content_proj = torch.load(args.content_proj, map_location="cpu")
    patch_proj = torch.load(args.patch_proj, map_location="cpu")
    content_size = content_proj["W"].shape[0]
    output_size, patch_dim = patch_proj["A"].shape
    cestimator = ContentEstimator(
        hidden_size=HIDDEN_SIZE,
        content_size=content_size,
        patch_size=patch_dim // content_size,
        output_size=output_size,
    )
    cestimator.content_proj.load_state_dict(content_proj)
    cestimator.patch_proj.load_state_dict(patch_proj)

    kmeans = None
    if args.kmeans is not None:
        checkpoint = torch.load(args.kmeans, map_location="cpu")
        kmeans = KMeans(*checkpoint["codebook.weight"].shape)
        kmeans.load_state_dict(checkpoint)
    return cestimator, kmeans


def train(args):
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)
    rank = dist.get_rank()
    logging.info(f"Start DDP training on rank {rank}.")

    encoder = Encoder()
    encoder.load_state_dict(torch.load(args.encoder, map_location="cpu"))
    encoder = encoder.to(rank).eval()
    encoder.requires_grad_(False)

    cestimator, kmeans = load_content_estimator(args)
    cestimator = cestimator.to(rank).eval()
    patch_size = cestimator.patch_proj.patch_size
    if kmeans is not None:
        kmeans = kmeans.to(rank).eval()
        num_codes = kmeans.codebook.num_embeddings
    if rank == 0:
        mode = f"tokens ({num_codes} clusters)" if kmeans is not None else "continuous content"
        logger.info(f"Training on {mode} with patch size {patch_size}")

    decoder = Decoder(
        content_channels=cestimator.patch_proj.A.shape[0],
        residual_channels=HIDDEN_SIZE,
        speaker_channels=SPEAKER_SIZE,
        **DECODER_CONFIGS[patch_size],
    )
    decoder = decoder.to(rank)
    finetune = kmeans is not None
    lr = FINETUNE_LR if finetune else PRETRAIN_LR
    max_steps = FINETUNE_STEPS if finetune else PRETRAIN_STEPS
    disc_model = Discriminator().to(rank)

    if args.init_from is not None:
        # Fine-tuning: start from a (continuous) pretrained checkpoint.
        checkpoint = torch.load(args.init_from, map_location="cpu")
        decoder.load_state_dict(checkpoint["decoder"])
        disc_model.load_state_dict(checkpoint["discriminator"])
        if rank == 0:
            logger.info(f"Initialized from {args.init_from}")

    generator = DDP(decoder, device_ids=[rank])
    discriminator = DDP(disc_model, device_ids=[rank])

    encoder = torch.compile(encoder)
    generator = torch.compile(generator)
    discriminator = torch.compile(discriminator)

    def extract_content(wavs: torch.Tensor):
        """Run the frozen front end. Returns (content, indices, residual)."""
        x = encoder(wavs)
        # Project and quantize in full precision so the tokens match inference.
        with torch.autocast("cuda", enabled=False):
            content, residual = cestimator(x.float())
            indices = None
            if kmeans is not None:
                indices, content = kmeans(content)
        return content, indices, residual

    recon_loss = MultiResolutionMelSpectralLoss().to(rank)

    opt_gen = optim.AdamW(
        generator.parameters(),
        lr=lr,
        betas=GEN_BETAS,
        weight_decay=WEIGHT_DECAY,
        eps=ADAM_EPS,
    )

    scheduler_gen = InverseLR(
        opt_gen,
        inv_gamma=LR_DECAY,
        power=LR_POWER,
        warmup=LR_WARMUP,
    )

    opt_disc = optim.AdamW(
        discriminator.parameters(),
        lr=lr,
        betas=DISC_BETAS,
        weight_decay=WEIGHT_DECAY,
        eps=ADAM_EPS,
    )

    scheduler_disc = InverseLR(
        opt_disc,
        inv_gamma=LR_DECAY,
        power=LR_POWER,
        warmup=LR_WARMUP,
    )

    log_dir = args.log_dir
    log_dir.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir) if rank == 0 else None

    train_dataset = AudioDataset(args.dataset_dir, train=True)
    if rank == 0:
        print(f"Train dataset has {len(train_dataset)} files")
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    valid_dataset = AudioDataset(args.dataset_dir, train=False)
    if rank == 0:
        print(f"Valid dataset has {len(valid_dataset)} files")

    valid_sampler = DistributedSampler(valid_dataset, shuffle=False, drop_last=True)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        sampler=valid_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    train_meter = LossMeter(win_length=LOG_INTERVAL)
    valid_meter = LossMeter(win_length=LOG_INTERVAL)

    global_step = 0
    if kmeans is not None:
        train_codes_used = torch.zeros(num_codes, dtype=torch.bool, device=rank)
        valid_codes_used = torch.zeros(num_codes, dtype=torch.bool, device=rank)

    checkpoint_loss = float("inf")
    epoch = 0
    while global_step < max_steps:
        train_meter.reset()
        train_sampler.set_epoch(epoch)

        generator.train()
        discriminator.train()

        for step, wavs in enumerate(train_loader):
            wavs = wavs.to(rank)

            with torch.autocast("cuda", torch.bfloat16):
                with torch.no_grad():
                    content, indices, residual = extract_content(wavs)
                    if indices is not None:
                        train_codes_used[indices.reshape(-1)] = True

                wavs_ = generator(content, residual)
                # Align the target with the decoder output: WavLM frames are
                # 400-sample windows with a 320-sample hop, so the decoded
                # audio starts 40 samples in and ends 40 samples early.
                wavs = wavs[..., 40:-40]

                (pscores_real, rscores_real), _ = discriminator(wavs)
                (pscores_fake, rscores_fake), _ = discriminator(wavs_.detach())

                pdisc_loss = hinge_loss(pscores_real, pscores_fake)
                rdisc_loss = hinge_loss(rscores_real, rscores_fake)
                disc_loss = pdisc_loss + rdisc_loss

            opt_disc.zero_grad()
            disc_loss.backward()
            clip_grad_norm_(discriminator.parameters(), max_norm=CLIP_GRAD_NORM)
            opt_disc.step()
            scheduler_disc.step()

            with torch.autocast("cuda", torch.bfloat16):
                with torch.no_grad():
                    _, (pfeatures_real, rfeatures_real) = discriminator(wavs)

                (pscores_fake, rscores_fake), (pfeatures_fake, rfeatures_fake) = (
                    discriminator(wavs_)
                )

                padv_loss = hinge_loss_adversarial(pscores_fake)
                radv_loss = hinge_loss_adversarial(rscores_fake)
                adv_loss = padv_loss + radv_loss
                pmatching_loss = feature_matching_loss(pfeatures_real, pfeatures_fake)
                rmatching_loss = feature_matching_loss(rfeatures_real, rfeatures_fake)
                matching_loss = pmatching_loss + rmatching_loss
                mel_loss = recon_loss(wavs_, wavs)
                gen_loss = adv_loss + MATCHING_WEIGHT * matching_loss + MEL_WEIGHT * mel_loss

            opt_gen.zero_grad()
            gen_loss.backward()
            clip_grad_norm_(generator.parameters(), max_norm=CLIP_GRAD_NORM)
            opt_gen.step()
            scheduler_gen.step()

            global_step += 1

            train_meter.update(
                {
                    "train/disc_loss": disc_loss.item(),
                    "train/pmatching_loss": pmatching_loss.item(),
                    "train/rmatching_loss": rmatching_loss.item(),
                    "train/matching_loss": matching_loss.item(),
                    "train/mel_loss": mel_loss.item(),
                    "train/adv_loss": adv_loss.item(),
                    "train/gen_loss": gen_loss.item(),
                },
            )

            if global_step % LOG_INTERVAL == 0:
                if kmeans is not None:
                    codes_used = train_codes_used.long()
                    dist.all_reduce(codes_used, op=dist.ReduceOp.MAX)
                    train_codes_used.zero_()

                if rank == 0:
                    for key, value in train_meter.windowed_average.items():
                        writer.add_scalar(key, value, global_step)
                    if kmeans is not None:
                        writer.add_scalar(
                            "train/codes_used", codes_used.sum().item(), global_step
                        )

            if global_step >= max_steps:
                break

        generator.eval()
        discriminator.eval()

        valid_meter.reset()
        if kmeans is not None:
            valid_codes_used.zero_()
        for step, wavs in enumerate(valid_loader):
            wavs = wavs.to(rank)
            with torch.autocast("cuda", torch.bfloat16):
                with torch.inference_mode():
                    content, indices, residual = extract_content(wavs)
                    if indices is not None:
                        valid_codes_used[indices.reshape(-1)] = True
                    wavs_ = generator(content, residual)
                    wavs = wavs[..., 40:-40]

                    (pscores_real, rscores_real), (pfeatures_real, rfeatures_real) = (
                        discriminator(wavs)
                    )
                    (pscores_fake, rscores_fake), (pfeatures_fake, rfeatures_fake) = (
                        discriminator(wavs_)
                    )

            pdisc_loss = hinge_loss(pscores_real, pscores_fake)
            rdisc_loss = hinge_loss(rscores_real, rscores_fake)
            disc_loss = pdisc_loss + rdisc_loss

            padv_loss = hinge_loss_adversarial(pscores_fake)
            radv_loss = hinge_loss_adversarial(rscores_fake)
            adv_loss = padv_loss + radv_loss
            pmatching_loss = feature_matching_loss(pfeatures_real, pfeatures_fake)
            rmatching_loss = feature_matching_loss(rfeatures_real, rfeatures_fake)
            matching_loss = pmatching_loss + rmatching_loss
            mel_loss = recon_loss(wavs_, wavs)
            gen_loss = adv_loss + MATCHING_WEIGHT * matching_loss + MEL_WEIGHT * mel_loss

            valid_meter.update(
                {
                    "valid/disc_loss": disc_loss.item(),
                    "valid/pmatching_loss": pmatching_loss.item(),
                    "valid/rmatching_loss": rmatching_loss.item(),
                    "valid/matching_loss": matching_loss.item(),
                    "valid/mel_loss": mel_loss.item(),
                    "valid/adv_loss": adv_loss.item(),
                    "valid/gen_loss": gen_loss.item(),
                },
            )

        totals, steps = valid_meter.running_totals
        keys = list(totals.keys())

        totals = torch.tensor(list(totals.values()), device=wavs.device)
        steps = torch.tensor(steps, device=wavs.device)

        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(steps, op=dist.ReduceOp.SUM)

        totals = {
            key: value / steps.item() for key, value in zip(keys, totals.tolist())
        }

        if kmeans is not None:
            valid_codes_used_long = valid_codes_used.long()
            dist.all_reduce(valid_codes_used_long, op=dist.ReduceOp.MAX)

        if rank == 0:
            for key, value in totals.items():
                writer.add_scalar(key, value, global_step)
            if kmeans is not None:
                writer.add_scalar(
                    "valid/codes_used", valid_codes_used_long.sum().item(), global_step
                )

            checkpoint_loss = totals["valid/mel_loss"]

            torch.save(
                {
                    "decoder": decoder.state_dict(),
                    "discriminator": disc_model.state_dict(),
                    "opt_gen": opt_gen.state_dict(),
                    "opt_disc": opt_disc.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                log_dir / f"epoch={epoch}_mel_loss={checkpoint_loss:.4f}.ckpt",
            )
        epoch += 1

    dist.destroy_process_group()
    logging.info(f"Finished running training on rank {rank}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the speaker embedder and decoder. Omit --kmeans to "
        "pretrain on continuous content; pass --kmeans and --init-from to "
        "fine-tune on tokens."
    )
    parser.add_argument(
        "dataset_dir",
        metavar="dataset-dir",
        help="path to the LibriSpeech directory (train-* and dev-* splits).",
        type=Path,
    )
    parser.add_argument(
        "log_dir",
        metavar="log-dir",
        help="directory for TensorBoard logs and checkpoints.",
        type=Path,
    )
    parser.add_argument(
        "--encoder",
        required=True,
        help="path to the WavLM encoder checkpoint (encoder.ckpt).",
        type=Path,
    )
    parser.add_argument(
        "--content-proj",
        required=True,
        help="path to the content projection checkpoint (content-proj.ckpt).",
        type=Path,
    )
    parser.add_argument(
        "--patch-proj",
        required=True,
        help="path to the PCA checkpoint (patch-proj-*.ckpt); sets the patch size.",
        type=Path,
    )
    parser.add_argument(
        "--kmeans",
        default=None,
        help="path to the codebook checkpoint (kmeans-*.ckpt). Omit to train on "
        "continuous content.",
        type=Path,
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="training checkpoint to initialize the speaker embedder, decoder and "
        "discriminator from (e.g. the continuous pretraining run).",
        type=Path,
    )
    args = parser.parse_args()
    logging.info("Starting decoder training")
    train(args)
