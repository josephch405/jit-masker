# Zoom Matting Project

This is a shared Pytorch repository between several teams working on efficient
portrait matting on video streams.

The main approach will be based on [JITNet](https://arxiv.org/pdf/1812.02699.pdf).

## Getting started

Install requirements with

```bash
pip install -r requirements
```

Alternatively, install them yourself with Conda somehow. You know the drill.

To get started, grab the [pretrained models for U-2-Net](https://github.com/NathanUA/U-2-Net#usage).
Either the full model or the small p-variant works - just put it in `/saved_models/u2net/u2net.pth`
or `/saved_models/u2netp/u2netp.pth`.

The main script to run is `python infer_video.py`. At the moment all the options
are contained in the file itself - we are working on streamlining the flow.
Comments within `infer_video.py` should be pretty explanatory.

At the moment you can choose between using a video, your own camera, or any IP-accessible
camera (for example, an Android phone with IP-webcam).

By default a video feed will be displayed. Check the script to see how to enable recording.

The script terminates whenever the stream terminates, or when the button q is pressed.

## Training

TODO

## Inference

TODO


## Shared modules

Models generally live in the [models](./models) folder.
[Dataloader](./data_loader.py)

