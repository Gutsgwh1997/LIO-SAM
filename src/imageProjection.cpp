/**
 * @file imageProjection.cpp
 * @brief 点云畸变补偿node 
 * @author GWH
 * @version 0.1
 * @date 2021-01-28 10:52:14
 * 1. cachePointCloud函数中，说明了对点云格式的要求！
 * 2. 此node主要进行畸变补偿，并且为了下个node更方便的提取点云特征，将点云的每条scan做了划分
 */
#include "utility.h"
#include "lio_sam/cloud_info.h"

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT 
{
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;                                      // 订阅原始imu测量
    std::deque<sensor_msgs::Imu> imuQueue;                       // 缓存虚拟imu坐标系下的imu测量

    ros::Subscriber subOdom;                                     // 订阅imu预积分node的高频lidar位姿
    std::deque<nav_msgs::Odometry> odomQueue;                    // 缓存高频lidar位姿（未与mapping模块对齐世界坐标系的）

    std::deque<sensor_msgs::PointCloud2> cloudQueue;             // 缓存原始雷达点云
    sensor_msgs::PointCloud2 currentCloudMsg;                    // ros格式的当前雷达点云

    double *imuTime = new double[queueLength];                   // imu帧时间戳 
    double *imuRotX = new double[queueLength];                   // imuDeskewInfo()函数内，使用陀螺仪原始测量进行欧拉积分
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;                                           // 当前帧点云采样时间内，imu帧的数量
    bool firstPointFlag;                                         // 该点是否为当前帧点云的起点
    Eigen::Affine3f transStartInverse;                           // 点云起点的仿射变换的逆

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;              // pcl格式的当前雷达点云
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;    // SensorType是OUSTER时的当前帧点云
    pcl::PointCloud<PointType>::Ptr   fullCloud;                 // 存储去畸变之后的原始点云
    pcl::PointCloud<PointType>::Ptr   extractedCloud;            // cloudExtraction()函数中，存储的与cloudInfo对应的点云

    int deskewFlag;                                              // 点云畸变补偿flag，采集的点云若没有time通道，则不会去畸变
    cv::Mat rangeMat;                                            // 原始点云转换为的深度图，图像像素存储对应位置的激光点深度

    bool odomDeskewFlag;                                         // imu预积分模块参与去畸变
    float odomIncreX;                                            // 使用imu预积分模块获得的lidar帧间平移
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;                                          // 当前帧点云起始点时间戳
    double timeScanEnd;                                          // 当前帧点云终止点时间戳
    std_msgs::Header cloudHeader;                                // 当前帧点云header


public:
    ImageProjection():
    deskewFlag(0)
    {
        // 接收原始imu测量
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        // 接收imu预积分node的高频lidar位姿
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // 接收原始lidr测量
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        // 这个话题只给rviz可视化使用
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        // 发布给下个node，cloudInfo中包含了extractedCloud信息
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    /**
     * @brief 为点云指针分配空间 
     */
    void allocateMemory() 
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    /**
     * @brief 重置点云与一些flag
     */
    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){}

    /**
     * @brief 原始imu测量的回调函数 
     *
     * @param imuMsg 原始imu测量
     */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg) 
    {
        // 将原始imu测量转换到虚拟imu坐标系下
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        /*
        {     
            cout << std::setprecision(6);
            cout << "IMU acc: " << endl;
            cout << "x: " << thisImu.linear_acceleration.x << 
                ", y: " << thisImu.linear_acceleration.y << 
                ", z: " << thisImu.linear_acceleration.z << endl;
            cout << "IMU gyro: " << endl;
            cout << "x: " << thisImu.angular_velocity.x << 
                ", y: " << thisImu.angular_velocity.y << 
                ", z: " << thisImu.angular_velocity.z << endl;
            double imuRoll, imuPitch, imuYaw;
            tf::Quaternion orientation;
            tf::quaternionMsgToTF(thisImu.orientation, orientation);
            tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
            cout << "IMU roll pitch yaw: " << endl;
            cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl; 
        }
        */
    }

    /**
     * @brief 接收imu预积分node的高频lidar位姿
     *
     * @param odometryMsg 高频lidar位姿（未与mapping模块对齐世界坐标系）
     */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    /**
     * @brief 这个类的主要处理函数
     *
     * @param laserCloudMsg 原始激光点云
     */
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 1.缓存ros格式的原始点云并转换为pcl格式，将缓存queue的首个元素作为当前雷达帧，同时检测点云是否有ring和time通道
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 2.点云畸变补偿预处理
        if (!deskewInfo())
            return;

        // 3.将原始采样点云映射为深度图，去除畸变后存储到fullCloud中
        projectPointCloud();

        // 4.转存激光点云每条线扫在一维数组中的开始和结束索引，以及其在rangeMat中的列id和range
        cloudExtraction();

        // 5.将本node处理过后的点云publish出去
        publishClouds();

        // 6.重置点云与一些flag
        resetParameters();
    }

    /**
     * @brief 缓存ros格式的原始点云并转换为pcl格式，检测点云是否有ring和time通道
     * 1. 若原始激光点云无time通道,不进行畸变补偿.
     *
     * @param laserCloudMsg 原始激光雷达测量
     *
     * @return 点云是否满足要求 
     */
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) 
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE) 
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        // TODO:: Apollo采集的数据集，这里要做出更改
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // check dense flag
        // TODO:: 移除NaN点
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /**
     * @brief 点云去畸变的预处理环节 
     */
    bool deskewInfo() 
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }
        // 使用陀螺仪原始测量在当前帧点云采样时间内进行欧拉积分
        imuDeskewInfo();
        // 利用imu预积分获得的高频lidar位姿，获得lidar帧间位移
        odomDeskewInfo();

        return true;
    }

    /**
     * @brief 使用陀螺仪原始测量在当前帧点云采样时间内进行欧拉积分 
     */
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                // 9轴imu的姿态
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            // 这里和上面多留出来0.01s，有利于findRotation()函数中对激光点云的起始点和终止点的插值
            // TODO:: 当imu频率较低时，建议适当增大时间
            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    /**
     * @brief 利用imu预积分模块预测的高频lidar位姿，获得lidar帧间位移 
     */
    void odomDeskewInfo() 
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    /**
     * @brief 获取激光点时刻相对于点云起点的欧拉角
     *
     * @param pointTime 此激光点的时间戳
     * @param rotXCur   相对于点云采样时刻，此激光点的roll
     * @param rotYCur   相对于点云采样时刻，此激光点的roll
     * @param rotZCur   相对于点云采样时刻，此激光点的roll
     */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur) 
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // 一帧点云的起始和终止阶段容易出现这种情况
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    /**
     * @brief 获取激光点时刻相对于点云起点的位移
     *
     * @param relTime  此激光点相对于点云起点的时间偏移
     * @param posXCur  相对于点云采样时刻，此激光点的x轴位移
     * @param posYCur  相对于点云采样时刻，此激光点的x轴位移
     * @param posZCur  相对于点云采样时刻，此激光点的x轴位移
     */
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur) {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    /**
     * @brief 将当前激光点投影回点云的起始时刻 
     *
     * @param point   当前激光点
     * @param relTime 当前激光点相对于点云起点的时间偏移
     *
     * @return 去除畸变后的点 
     */
    PointType deskewPoint(PointType *point, double relTime) 
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;

        // 获取激光点时刻该点相对于点云起点的欧拉角
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        // 获取激光点时刻该点相对于点云起点的位移，此函数暂时没用到！
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    /**
     * @brief 将原始采样点云映射为深度图，去除畸变后存储到fullCloud中 
     */
    void projectPointCloud() 
    {
        // pcl形式的当前帧点云
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            // 计算与y轴的夹角，范围(-PI,PI]
            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

            // 将点云投影为Horizon_SCAN*N_SCAN大小的深度图，点云的x轴投影到图像中心轴
            // |                  |
            // |-x  -y  x   y   -x|
            // |                  |
            static float ang_res_x = 360.0 / float(Horizon_SCAN);
            int columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // rangeMat和fullCloud存储顺序是保持一致的
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    /**
     * @brief 转存激光点云每条线扫在一维数组中的开始和结束索引，以及其在rangeMat中的列id和range 
     */
    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    /**
     * @brief 将本node处理过后的点云publish出去
     */
    void publishClouds() 
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    ros::MultiThreadedSpinner spinner(3);
    spinner.spin();
    
    return 0;
}
