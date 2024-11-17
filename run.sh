bin/replica_rgbd \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/RGB-D/Replica/office0.yaml \
    cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml \
    PATH_TO_Replica/office0 \
    PATH_TO_SAVE_RESULTS

bin/tum_mono \
    third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
    cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml \
    cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml \
    /data/TUM/dataset_freiburg1_desk \
    results/tum_mono/$exp/tum_mono_$i/rgbd_/dataset_freiburg1_desk \
    no_viewer

python3 python/eval.py --dataset_center_path /data --result_main_folder /workspaces/cartgs/results --exp_name test