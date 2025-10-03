# DACVAE
VAE version of the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec), which has a continuous latent space.
Descript Audio Codec (DAC) is a high fidelity general neural audio codec, introduced in the paper titled **High-Fidelity Audio Compression with Improved RVQGAN**.
Most code is adopted from the open-source repo [DAC](https://github.com/descriptinc/descript-audio-codec)

### Installation
```bash
$ pip install git+https://github.com/facebookresearch/dacvae
```

### Usage

```python
from dacvae import DACVAE
import torchaudio


model = DACVAE.load("facebook/dacvae-watermarked")
wav, sample_rate = torchaudio.load("<path to audio file>")
# Resample to expected sample rate
resampled = torchaudio.functional.resample(wav, sample_rate, model.sample_rate)
# Convert stero to mono (if applicable)
resampled = resampled.mean(dim=0, keepdim=True)
# Expected shape is batch x 1 x samples
model_input = resampled.unsqueeze(0)
encoded = model.encode(model_input)
# `decoded` shape is `batch x 1 x samples`
decoded = model.decode(encoded)
```

## Watermarking

The DAC-VAE decoder has been integrated with [Audioseal](https://github.com/facebookresearch/audioseal) to ensure all audios generated
contain watermarks that are verifiable independently. We develop a new watermarking model with adapted architecture specifically for
DAC-VAE to optimize the high-fidelity outcome. We also plan to release the detector API. Stay tuned !

## Contributing

See [contributing](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md) for more information.

## License

This project is licensed under the SAM License - see the [LICENSE](LICENSE) file for details.
