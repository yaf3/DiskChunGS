bin/replica_rgbd \
    slam_deps/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    /data/Replica/office0 \
    results/test


python3 python/eval.py --dataset_center_path /data --result_main_folder /workspaces/cartgs/results --exp_name test
