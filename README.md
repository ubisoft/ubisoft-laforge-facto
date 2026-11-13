Copyright Ubisoft(c) 2026
# Factorized Self-Supervised Speech Tokenization

**Benjamin van Niekerk**<sup>1,2,3</sup>, **Jean-Philippe Letendre**<sup>1</sup>, **Nicol Visser**<sup>4</sup>, **Hugo Seuté**<sup>1</sup>, **Herman Kamper**<sup>4</sup>, **Mirco Ravanelli**<sup>2,3</sup>

<sup>1</sup>Ubisoft La Forge, <sup>2</sup>Concordia University, <sup>3</sup>Mila, <sup>4</sup>Stellenbosch University

## Abstract

Speech tokenizers convert audio into discrete units, enabling language models to process and generate speech. The goal is to learn compact, text-like representations while preserving enough detail to reconstruct a speaker's voice and delivery. We propose Facto: a factorized tokenizer that disentangles time-varying information (content and prosody) from static components (speaker identity and recording conditions). We build on a recent linear factorization method for self-supervised features, extending it to handle unseen speakers. Then, we cluster the factorized space and train a lightweight decoder to reconstruct audio from the resulting tokens. By removing the static components before clustering, our tokens are more robust to small noise perturbations. We validate this by testing token stability under various noise conditions. Next, we theoretically analyze Facto's convergence and scaling properties, quantifying how the factorization improves with the number of training speakers. Finally, we evaluate Facto on a range of downstream tasks. We show that greater token stability translates into improved text-to-speech performance and competitive results in voice conversion and spoken language modeling.

## Inference

Pretrained models are available through `torch.hub`. Each model comes in four variants, named by frame rate and codebook size:

| Variant | Frame rate | Codebook size |
|---|---|---|
| `12hz-8192` (default) | 12.5 Hz | 8192 |
| `12hz-16384` | 12.5 Hz | 16384 |
| `25hz-8192` | 25 Hz | 8192 |
| `25hz-16384` | 25 Hz | 16384 |

The tokenizers take 16 kHz mono audio as input. The examples below use [torchcodec](https://github.com/pytorch/torchcodec) to load audio.

### Text-to-speech

The TTS model generates speech given a text prompt and a short reference clip (along with its transcript). The `tts` entry requires `transformers`.

```python
import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

facto = torch.hub.load("ubisoft/ubisoft-laforge-facto", "facto", variant="12hz-8192", trust_repo=True).cuda()
lm, tokenizer = torch.hub.load("ubisoft/ubisoft-laforge-facto", "tts", variant="12hz-8192", trust_repo=True)
lm = lm.cuda()

reference = AudioDecoder("reference.wav", sample_rate=16000, num_channels=1)
reference = reference.get_all_samples().data.unsqueeze(0).cuda()

with torch.inference_mode():
    # Tokenize the reference clip and extract its speaker embedding.
    reference_tokens, speaker = facto.encode(reference)

    # Build the prompt: reference transcript, target text, and reference tokens.
    input_ids, attention_mask = tokenizer.encode(
        [("Transcript of the reference clip.", "Text to synthesize.", reference_tokens[0])]
    )
    output_ids = lm.generate(
        input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        do_sample=True,
        temperature=0.5,
        top_k=0,
        repetition_penalty=1.1,
        max_new_tokens=375,  # 30 s at 12.5 Hz (use 750 for 25 Hz)
        cache_implementation="dynamic",  # faster for one-off generations
    )
    tokens = tokenizer.decode(output_ids, input_ids.shape[1])[0]

    # Decode the generated tokens in the reference speaker's voice.
    audio = facto.decode(tokens.unsqueeze(0), speaker)

AudioEncoder(audio.squeeze(0).cpu(), sample_rate=16000).to_file("output.wav")
```

The same steps are available as a script:

```bash
python scripts/tts.py reference.wav "Transcript of the reference clip." "Text to synthesize." output.wav --variant 12hz-8192
```

### Voice conversion

Voice conversion combines the content tokens of a source utterance with the target speaker embedding.

```python
import torch
from torchcodec.decoders import AudioDecoder
from torchcodec.encoders import AudioEncoder

facto = torch.hub.load("ubisoft/ubisoft-laforge-facto", "facto", variant="12hz-8192", trust_repo=True).cuda()

def load(path):
    decoder = AudioDecoder(path, sample_rate=16000, num_channels=1)
    return decoder.get_all_samples().data.unsqueeze(0).cuda()

with torch.inference_mode():
    tokens, _ = facto.encode(load("source.wav"))   # what is said
    _, speaker = facto.encode(load("target.wav"))  # who says it
    audio = facto.decode(tokens, speaker)

AudioEncoder(audio.squeeze(0).cpu(), sample_rate=16000).to_file("converted.wav")
```

The same steps are available as a script:

```bash
python scripts/vc.py source.wav target.wav converted.wav --variant 12hz-8192
```

## Training

Training Facto consists of four stages: 1. encoding the dataset with WavLM, 2. factorizing the features and fitting the content estimator, 3. clustering the content space, and 4. training the decoder. The factorization, content estimation, and clustering run on a CPU. The decoder takes the bulk of the training time and requires at least one GPU.

### 1. Install

Clone the repository and install the package:

```bash
git clone https://github.com/ubisoft/ubisoft-laforge-facto.git
cd ubisoft-laforge-facto
pip install -e .
```

Download [LibriSpeech](https://www.openslr.org/12) and the pretrained WavLM encoder (WavLM-Large truncated to layer six):

```bash
wget https://github.com/ubisoft/ubisoft-laforge-facto/releases/download/v0.1.0/encoder-2bc62958.ckpt -O encoder.ckpt
```

### 2. Encode LibriSpeech

Extract WavLM features for the `train-clean-100` and `train-clean-360` sets:

```bash
python scripts/encode.py LibriSpeech/train-clean-100 features/train-clean-100 --encoder encoder.ckpt
python scripts/encode.py LibriSpeech/train-clean-360 features/train-clean-360 --encoder encoder.ckpt
```

The features are saved as 1024-dimensional float16 arrays at 50 Hz, which takes roughly 0.37 GB per hour of audio (about 170 GB for `train-clean`).

### 3. Factorize and fit the content estimator

Factorize `train-clean-100` and fit the least-squares content estimator:

```bash
python scripts/factorize.py features/train-clean-100 checkpoints
```

This samples five minutes of speech per speaker, factorizes the aligned features with a content rank of 12, and saves the content projection to `checkpoints/content-proj.ckpt`.

### 4. Cluster the content space

Project `train-clean` into the content space, stack neighboring frames into patches, apply PCA, and cluster the result with k-means:

```bash
# 12.5 Hz (patches of 4 frames)
python scripts/cluster.py features checkpoints/content-proj.ckpt checkpoints --patch-size 4 --num-clusters 8192

# 25 Hz (patches of 2 frames)
python scripts/cluster.py features checkpoints/content-proj.ckpt checkpoints --patch-size 2 --num-clusters 8192
```

This saves the PCA projection (`patch-proj-12hz.ckpt` or `patch-proj-25hz.ckpt`) and the codebook (e.g. `kmeans-12hz-8192.ckpt`). The PCA depends only on the frame rate, so different codebook sizes share it.

### 5. Train the decoder

The decoder is trained on the full `train-960` set (the `train-*` splits of LibriSpeech, with the `dev-*` splits used for validation). Training has two stages. First, pretrain on continuous content, which takes 2M steps:

```bash
torchrun --nproc-per-node 2 scripts/train_decoder.py LibriSpeech logs/12hz-cont \
    --encoder encoder.ckpt \
    --content-proj checkpoints/content-proj.ckpt \
    --patch-proj checkpoints/patch-proj-12hz.ckpt
```

Then fine-tune on the tokens, starting from the pretrained checkpoint, which takes 500K steps:

```bash
torchrun --nproc-per-node 2 scripts/train_decoder.py LibriSpeech logs/12hz-8192 \
    --encoder encoder.ckpt \
    --content-proj checkpoints/content-proj.ckpt \
    --patch-proj checkpoints/patch-proj-12hz.ckpt \
    --kmeans checkpoints/kmeans-12hz-8192.ckpt \
    --init-from logs/12hz-cont/<checkpoint>.ckpt
```


Passing `--kmeans` switches to the fine-tuning settings. The same pretrained decoder can initialize fine-tuning runs with different codebook sizes. Checkpoints and TensorBoard logs are written to the log directory, and the decoder weights are stored under the `decoder` key of each checkpoint.

### License 
Please check license in ./license folder. 
./license/LICENSE covering this project in general :   Ubisoft Machine Learning License

./license/MONAD_LICENCE covering :  
All variants of the TTS model available in this repository are fine-tuned derivatives of the PleIAs/Monad model found here: https://huggingface.co/PleIAs/Monad, which is licensed under the Apache License 2.0.
Modifications include updated model weights and additional entries in the tokenizer vocabulary.
Please refer to the original license (found under MONAD_LICENCE in this repository) for more details on usage restrictions and permissions of the PleIAs/Monad model. 


Copyright Ubisoft(c) 2026
