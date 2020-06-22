# JIT-Masker

\[[arxiv](https://arxiv.org/abs/2006.06185)\]

## Getting started

Install requirements with

```bash
pip install -r requirements
```

Install detectron separately with

```bash
pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

To use U2Net, grab the [pretrained models for U-2-Net](https://github.com/NathanUA/U-2-Net#usage).
Either the full model or the small p-variant works - just put it in `/saved_models/u2net/u2net.pth`
or `/saved_models/u2netp/u2netp.pth`.

A pretrained JITNet model for this repository can be found [here](https://drive.google.com/file/d/118QfdHhd8KmoZx6MgxO5xAtBHzgfVW71/view?usp=sharing). Put this model at `/saved_models/jitnet/jitnet.pth`.

The main script to run is `infer_video.py`. For example, to run JIT-Masker
with your laptop webcam, run `python infer_video.py -i 0`.

At the moment you can choose between using a video, your own camera, or any IP-accessible
camera (for example, an Android phone with IP-webcam).

The script terminates whenever the stream terminates, or when the button q is pressed.

## Training (Optional)

Run `python train.py` to train a JITNet model on the Supervisely dataset. The
Supervisely dataset is available on their [official website](https://app.supervise.ly/login)
and should be downloaded to `data/supervisely`.

## Inference

