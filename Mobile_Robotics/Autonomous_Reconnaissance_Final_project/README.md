# Autonomous Reconnaissance for Disaster Relief 

## EECE-5550 Mobile Robotics Project @ Northeastern University 

Disaster relief efforts can be dangerous for human
rescuers. Rescuers may need to perform exhaustive searches of
disaster areas when searching for victims, exposing themselves
to life-threatening hazards. This makes disaster reconnaissance an 
ideal candidate for automatization.

Autonomous reconnaissance requires a robotic agent to optimally
search the disaster area and identify victims. This identification
provides an entry point of rescue for human intervention. In our
setup, we simulate a disaster environment using a Turtlebot as
an autonomous rescue agent, and AprilTags to represent victims.
We propose a customized searching and mapping algorithm
that attempts to minimize rescue times and prevents search
termination before all victims are identified. In this approach, we
repeatedly map the disaster environment using the explore lite
ROS package. After each complete mapping, explore lite is
reinitialized with the robot starting in the least-explored region
of the map. This reinitialization is determined using a heatmap
generated from the path the robot follows during the previous
mapping iteration. We simulate this disaster rescue scenario
problem in our setup and assess how variations of our approach
detect and localize AprilTags.


## Setup and Procedure

* Place the robot in an unknown environment.
* We start Gmapping using SLAM, to map the environment.
* Also start the explore_lite node parallely to start the bot to move autonomously in the environment.
* Detect AprilTags and store their global poses.
* At the same time, /odom and /map topics are saved in a rosbag
* The path traced by the robot is extracted from pose.x and pose.y from /odom topic and the map is extracted from /map topic.
* This data is used to generate a heatmap based on a prioritization strategy, based on the robot path, detected AprilTag poses, or obstacles.
* From this heatmap, a new reinitialization point is determined so as to increase the number of AprilTags detected in successive iterations.

## References
 * [mr_final_project](https://github.com/lhy0807/mr_final_project)
