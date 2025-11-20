# Additional Information

This document contains additional utilities and information for working with DiskChunGS.

## Hypertuning Scripts

There exist hypertuning scripts `scripts/hypertune_kitty.py` and `scripts/hypertune_rsl.py`. These are pretty much thrown together and you will have to change the paths/parameters inside these scripts.

## Dataset Tools

### Rosbag Extractor

`rosbag_extractor.py` can be used to generate TUM style datasets from rosbags. 

**Usage:**
- It will play the bag but you will have to press space after running the script
- It can also work with multiple bags if you add all their paths as arguments

### Video Creation

`img2vid.py` can be used to create a video from images.

## Utility Scripts

Various utility scripts are available in the repository for different purposes:

- **Dataset download scripts**: `scripts/download_replica.sh`, `scripts/download_tum.sh`
- **Experiment scripts**: `scripts/replica_mono.sh`, `scripts/replica_rgbd.sh`, `scripts/tum_mono.sh`, `scripts/tum_rgbd.sh`, `scripts/kitti_stereo.sh`
- **Hypertuning scripts**: `scripts/hypertune_kitty.py`, `scripts/hypertune_rsl.py`
- **Dataset extraction**: `rosbag_extractor.py`
- **Video generation**: `img2vid.py`

## Notes

- Most utility scripts require manual path adjustments
- Hypertuning scripts are experimental and need customization
- Always verify paths and parameters before running scripts