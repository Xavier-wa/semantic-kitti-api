import numpy as np
from PIL import Image

def readPointCloudAndLabel(pcFile):
    raw_data = np.fromfile(pcFile, dtype=np.float32).reshape((-1, 4))
    origin_len = len(raw_data)
    points = raw_data[:, :3]
    annotated_data = np.fromfile(pcFile.replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
    indices = np.argwhere(annotated_data[:,0]==40)
    raw_data=raw_data[indices]
    annotated_data=annotated_data[indices]
    raw_data.tofile("/mnt/storage/xhc/LidarSeg/semantic-kitti-api/dataset/SemanticKittiDir/dataset/sequences/000000.bin")
    annotated_data.tofile("/mnt/storage/xhc/LidarSeg/semantic-kitti-api/dataset/SemanticKittiDir/dataset/sequences/000000.label")

if __name__ == "__main__":
    readPointCloudAndLabel("/mnt/storage/xhc/LidarSeg/semantic-kitti-api/dataset/SemanticKittiDir/dataset/sequences/00/velodyne/000000.bin")
