# DiskChunGS: Memory-Unbounded 3D Gaussian SLAM Through Efficient Disk Chunking
[Casimir Feldmann](https://scholar.google.com/citations?user=WNqWurwAAAAJ&hl=en&oi=ao)<sup>1</sup>, [Max Wilder-Smith](https://github.com/maxwildersmith)<sup>1</sup>, [Vaishakh Patil](https://scholar.google.com/citations?user=aB04078AAAAJ&hl=en)<sup>1</sup>, [Michael Niemeyer](https://m-niemeyer.github.io/)<sup>2</sup>, [Michael Oechsle](https://moechsle.github.io/)<sup>2</sup>, [Keisuke Tateno](https://scholar.google.com/citations?user=ml3laqEAAAAJ&hl=ja)<sup>2</sup>, and [Marco Hutter](https://scholar.google.ch/citations?user=DO3quJYAAAAJ&hl=en)<sup>1</sup> <br>
ETH Zurich<sup>1</sup>, Google<sup>2</sup>
<br>
[[`Paper`]()] [[`Project`]()] [[`Demo`]()] [[`Dataset`]()] [[`BibTeX`]()]

![Pipeline](assets/main_pipeline_resized.png?raw=true)

DiskChunGS is a 3D Gaussian Splatting SLAM system that enables unbounded scene reconstruction through dynamic memory management, partitioning environments into spatial chunks that are selectively loaded between GPU and disk storage. This innovative approach achieves substantially higher Gaussian density than previous methods, resulting in significantly improved reconstruction quality across diverse environments while maintaining real-time performance.

## Installation

Installation herer (docker image)

## Getting Started

Usage guide here and eval info


## Common Issues:

- If you get "ImportError: Cannot load backend 'Qt5Agg' which requires the 'qt' interactive framework, as 'headless' is currently running" during eval then make sure the display is forwarded. Run "xhost +local:root"

## Acknowledgement
This work incorporates many open-source codes. Thanks for their great work!
- [CaRtGS](https://github.com/DapengFeng/cartgs)
- [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM)
- [Taming 3DGS](https://github.com/humansensinglab/taming-3dgs)
- [Frustum Culling](https://bruop.github.io/frustum_culling/)

# Citation
If you find this work useful in your research, consider citing it:
```

```