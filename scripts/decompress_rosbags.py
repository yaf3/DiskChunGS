#!/usr/bin/env python3
import os
import errno
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

ids = {"apartment_scene":
    ["uHumans2_apartment_s1_00h.bag",
     "uHumans2_apartment_s1_01h.bag",
     "uHumans2_apartment_s1_02h.bag"
    ],
 "office_scene":
    ["uHumans2_office_s1_00h.bag",
     "uHumans2_office_s1_06h.bag",
     "uHumans2_office_s1_12h.bag"
    ],
 "subway_scene":
    [
     "uHumans2_subway_s1_00h.bag",
     "uHumans2_subway_s1_24h.bag",
     "uHumans2_subway_s1_36h.bag"
    ],
 "neighborhood_scene":
    [
     "uHumans2_neighborhood_s1_00h.bag",
     "uHumans2_neighborhood_s1_24h.bag",
     "uHumans2_neighborhood_s1_36h.bag"
    ]
}

def decompress_bag(dataset_dir, dataset_name, rosbag_name):
    """Decompress a single rosbag file"""
    full_path = os.path.join(dataset_dir, dataset_name, rosbag_name)
    print(f"Starting decompression of rosbag: {rosbag_name}")
    result = os.system(f"rosbag decompress {full_path}")
    print(f"Done decompressing rosbag: {rosbag_name}")
    return (rosbag_name, result)

def run(args):
    assert(os.path.exists(args.dataset_dir))
    
    # Create a list of all decompression tasks
    tasks = []
    for dataset_name in ids.keys():
        dataset_path = os.path.join(args.dataset_dir, dataset_name)
        assert(os.path.exists(dataset_path))
        for rosbag_name in ids[dataset_name]:
            tasks.append((args.dataset_dir, dataset_name, rosbag_name))
    
    # Determine the number of workers (use max(1, CPU count - 1) to leave one CPU free)
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Decompressing {len(tasks)} rosbags using {max_workers} workers")
    
    # Use ProcessPoolExecutor to parallelize the decompression
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_bag = {
            executor.submit(decompress_bag, dataset_dir, dataset_name, rosbag_name): (dataset_name, rosbag_name)
            for dataset_dir, dataset_name, rosbag_name in tasks
        }
        
        # Process results as they complete
        for future in as_completed(future_to_bag):
            dataset_name, rosbag_name = future_to_bag[future]
            try:
                bag_name, result = future.result()
                if result != 0:
                    print(f"Warning: decompression of {bag_name} may have failed with exit code {result}")
            except Exception as e:
                print(f"Error decompressing {rosbag_name} in {dataset_name}: {e}")
    
    print("Done decompressing all rosbags.")
    return True

def parser():
    import argparse
    basic_desc = "Decompress uHumans2 dataset."
    shared_parser = argparse.ArgumentParser(add_help=True, description="{}".format(basic_desc))
    output_opts = shared_parser.add_argument_group("output options")
    output_opts.add_argument(
        "--dataset_dir", type=str, help="Path to the directory where the datasets are.", required=True)
    main_parser = argparse.ArgumentParser(description="{}".format(basic_desc))
    sub_parsers = main_parser.add_subparsers(dest="subcommand")
    sub_parsers.required = True
    return shared_parser

if __name__ == '__main__':
    import argcomplete
    import sys
    parser = parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    if run(args):
        sys.exit(os.EX_OK)