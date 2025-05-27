# Troubleshooting

This document covers common issues and their solutions when using DiskChunGS.

## Display and Visualization Issues

### Qt Backend Error
**Problem**: `ImportError: Cannot load backend 'Qt5Agg' which requires the 'qt' interactive framework, as 'headless' is currently running` during eval

**Solution**: Make sure the display is forwarded. Run `xhost +local:root`

## System Crashes and Errors

### Configuration Related Crashes
**Problem**: System crashes or exhibits weird behavior

**Solution**: First look for mistakes in the configuration files. If they are misconfigured, DiskChunGS will not validate them and exhibit weird behavior or crashing.

### Memory Issues
**Problem**: System runs out of memory

**Solution**: 
- Reduce the number of `Chunking.max_chunks`
- For large scenes, use `Mapper.keyframe_selection_strategy` set to `1`

### Dependency Issues
**Problem**: Weird errors in the dependencies

**Solution**: Try running `./clean.sh` and building fresh.

## ROS Related Issues

### ROS Commands Not Working
**Problem**: ROS commands fail to execute

**Solution**: Run `source /opt/ros/noetic/setup.bash`

### RSL RGB-D Script Issues
**Problem**: The `rsl_rgbd.sh` script does not play rosbags automatically

**Solution**: 
- Rosbags have to be played manually by the user
- You will also have to publish uncompressed topics

## Evaluation Issues

### Python Wrapper Segmentation Fault
**Problem**: Segmentation fault in the python wrapper after evaluation

**Solution**: This is a known issue but doesn't affect the evaluation results or functionality.

## General Tips

- Always check configuration files first when encountering issues
- Ensure proper display forwarding for visualization components
- Monitor memory usage, especially with large scenes
- Verify that all required dependencies are properly built and sourced