# Cloud Info
Header header 

int32[] startRingIndex  # 激光点云存储为一维，标记每条线扫的开始索引 
int32[] endRingIndex    # 激光点云存储为一维，标记每条线扫的结束索引

int32[]  pointColInd    # point column index in range image
float32[] pointRange    # point range 

int64 imuAvailable
int64 odomAvailable

# Attitude for LOAM initialization which get from a 9-axis IMU
float32 imuRollInit
float32 imuPitchInit
float32 imuYawInit

# Initial guess from imu pre-integration
float32 initialGuessX
float32 initialGuessY
float32 initialGuessZ
float32 initialGuessRoll
float32 initialGuessPitch
float32 initialGuessYaw

# Point cloud messages
sensor_msgs/PointCloud2 cloud_deskewed  # original cloud deskewed
sensor_msgs/PointCloud2 cloud_corner    # extracted corner feature
sensor_msgs/PointCloud2 cloud_surface   # extracted surface feature
