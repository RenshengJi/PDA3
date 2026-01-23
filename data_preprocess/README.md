# Data Preprocessing for Physical AI AV Dataset

This directory contains scripts for downloading and preparing the Physical AI AV dataset for training.

**Note**: The Hugging Face token is only needed for downloading. Once data is downloaded, training does NOT require a token.

## Prerequisites

1. Install the physical_ai_av package:
```bash
pip install physical_ai_av
```

2. Set up your Hugging Face token:
```bash
huggingface-cli login
```

Or export it directly:
```bash
export HF_TOKEN="your_token_here"
```

## Download Dataset

The `download_physical_ai_av.py` script downloads camera images, LiDAR point clouds, and calibration data from NVIDIA's PhysicalAI-Autonomous-Vehicles dataset.

### Basic Usage

Download first 100 clips with all cameras:
```bash
python download_physical_ai_av.py \
    --output_dir ./data \
    --max_clips 100 \
    --cameras all \
    --download_lidar
```

### Command Line Arguments

- `--output_dir`: Directory to save downloaded data (default: `./data`)
- `--token`: Hugging Face token (default: uses cached token from `huggingface-cli login`)
- `--max_clips`: Maximum number of clips to download (default: all available)
- `--start_clip`: Start downloading from this clip index (default: 0)
- `--cameras`: Camera selection:
  - `all`: All 7 cameras (default)
  - `front`: Only front cameras (front_wide_120fov, front_tele_30fov)
  - `custom`: Specify custom list with `--custom_cameras`
- `--custom_cameras`: List of camera names (only with `--cameras=custom`)
- `--download_lidar`: Download LiDAR data (default: True)

### Available Cameras

- `camera_front_wide_120fov` - Front wide-angle (120° FOV)
- `camera_front_tele_30fov` - Front telephoto (30° FOV)
- `camera_rear_tele_30fov` - Rear telephoto (30° FOV)
- `camera_cross_left_120fov` - Left cross-view (120° FOV)
- `camera_cross_right_120fov` - Right cross-view (120° FOV)
- `camera_rear_left_70fov` - Rear left (70° FOV)
- `camera_rear_right_70fov` - Rear right (70° FOV)

## Dataset Structure

After downloading, your data directory will have this structure:

```
data/
├── clip_index.parquet                    # Clip metadata
├── features.csv                          # Feature definitions
├── metadata/
│   └── sensor_presence.parquet           # Sensor availability per clip
├── camera/
│   ├── camera_front_wide_120fov/
│   │   ├── camera_front_wide_120fov.chunk_0000.zip
│   │   ├── camera_front_wide_120fov.chunk_0001.zip
│   │   └── ...
│   ├── camera_front_tele_30fov/
│   │   └── camera_front_tele_30fov.chunk_XXXX.zip
│   └── ... (other cameras)
├── lidar/
│   └── lidar_top_360fov/
│       ├── lidar_top_360fov.chunk_0000.zip
│       ├── lidar_top_360fov.chunk_0001.zip
│       └── ...
└── calibration/
    ├── camera_intrinsics/
    │   ├── camera_intrinsics.chunk_0000.parquet
    │   └── ...
    ├── sensor_extrinsics/
    │   ├── sensor_extrinsics.chunk_0000.parquet
    │   └── ...
    └── vehicle_dimensions/
        └── vehicle_dimensions.chunk_XXXX.parquet
```

**Note**: Each feature (camera, lidar, calibration) is organized in its own directory, with chunk files named as `{feature_name}.chunk_{chunk_id}.{ext}`. Multiple clips are packed into each chunk file.



### HF_ENDPOINT for China users
The script automatically sets `HF_ENDPOINT='https://hf-mirror.com'`. If you need a different mirror, edit the script's first line.
