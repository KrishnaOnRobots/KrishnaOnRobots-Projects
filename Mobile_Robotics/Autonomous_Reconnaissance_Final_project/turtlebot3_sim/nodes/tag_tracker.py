#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs
import queue
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import TransformStamped
from visualization_msgs.msg import MarkerArray, Marker

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import numpy as np
import os
import time
start_time = time.time()

from geometry_msgs.msg import Twist
PI = 3.1415926535897
if os.path.exists("/home/ahmet/catkin_ws/src/mr_final_project/nodes/tag_recent_locations.txt"):
    os.remove("/home/ahmet/catkin_ws/src/mr_final_project/nodes/tag_recent_locations.txt")
else:
    print("The file does not exist")
file1 = open("/home/ahmet/catkin_ws/src/mr_final_project/nodes/tag_recent_locations.txt","a")

if os.path.exists("/home/ahmet/catkin_ws/src/mr_final_project/nodes/tag_detect.txt"):
  os.remove("/home/ahmet/catkin_ws/src/mr_final_project/nodes/tag_detect.txt")
else:
  print("The file does not exist")
file2 = open("/home/ahmet/catkin_ws/src/mr_final_project/nodes/tag_detect.txt","a")

if os.path.exists("/home/ahmet/catkin_ws/src/mr_final_project/nodes/robot_locations.txt"):
    os.remove("/home/ahmet/catkin_ws/src/mr_final_project/nodes/robot_locations.txt")
else:
    print("The file does not exist")
robot_locations = open("/home/ahmet/catkin_ws/src/mr_final_project/nodes/robot_locations.txt","a")
class TagsUpdater:

    def __init__(self):
        rospy.sleep(1.)
        num_tags = 20
        self.STABLE_SIZE=10
        self.tag_estimates = [queue.Queue(self.STABLE_SIZE) for _ in range(num_tags)]
        self.tags_sub = rospy.Subscriber("tag_detections", \
            AprilTagDetectionArray, self.tags_cb)
        #rospy.init_node('odom_node')   
        self.odom = rospy.Subscriber('/odom', Odometry, self.callback)
        #sub = rospy.Subscriber('/odom', Odometry, self.callback)
        # Spin until the node is shutdown
        self.just_rotated = False
        
        self.vis_pub = rospy.Publisher("visualization_marker", Marker, \
            queue_size=1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, queue_size=1)
        self.vis_pub = rospy.Publisher("visualization_marker", Marker, \
            queue_size=1)
        self.most_recent_locations = np.zeros((3, 6))
        self.detect_time = np.zeros((1,6))
        
        while True:
            try:
                _ = self.tf_buffer.lookup_transform('map', 'base_link', \
                    rospy.Time(0))
                break
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, \
                    tf2_ros.ExtrapolationException):
                continue

    
        
    def callback(self, msg):
        # Print the robot's current pose
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler = euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]
        robot_locations.write("%.3f " % float(x) )
        robot_locations.write("%.3f " % float(y) )
        robot_locations.write("%.3f " % float(yaw) )
        robot_locations.write("\n")
#        robot_locations.write("%.3f " % float(x) )
#        robot_locations.write("%.3f " % float(y) )
#        robot_locations.write("%.3f " % float(yaw) )
#        robot_locations.write("\n")
        print("Robot's current pose: (%.2f, %.2f, %.2f)", x, y, yaw)
        #print("TIME TIME", float(rospy.get_time()) )
        if float(rospy.get_time()) % 1 < 0.5 and not self.just_rotated:
            print("rotate")
            #rospy.sleep(1)
            rospy.spin()
        #    rotate()
            self.just_rotated = True
        #    #rospy.sleep(1)
            print("TIME", float(rospy.get_time()))
        else:
            self.just_rotated = False
            print("ELSE", float(rospy.get_time()))
	    

    def tags_cb(self, tags):
        # iterate through tags
        #print("DETECTED TAGS", len(tags.detections))
        for tag in tags.detections:
            # Get tag info
            tag_id = tag.id[0]

            # do not update the tag pose if detected before
            # if self.tag_estimates[tag_id]:
            #     continue

            tag_name = 'map_tag_' + str(tag_id)
            frame_tag_pose = tag.pose.pose.pose
            frame = tag.pose.header.frame_id

            # Get transform from map to tag
            map_tag_pose = self.transform_pose(frame_tag_pose, frame, 'map')
            if (len(self.tag_estimates[tag_id].queue) >= self.STABLE_SIZE-1):
                self.tag_estimates[tag_id].get()
            
            self.tag_estimates[tag_id].put(map_tag_pose, block=False)

        for i, est in enumerate(self.tag_estimates):
            if not est.empty():
                position = np.zeros((3, len(self.tag_estimates[tag_id].queue)))
                for j, pose in enumerate(self.tag_estimates[i].queue):
                    if pose is None:
                        continue
                    position[0][j] = pose.position.x
                    position[1][j] = pose.position.y
                    position[2][j] = pose.position.z
                
                centroid_pose = np.mean(position, axis=1)
                msg = "tag" + str(i) +  ": "+  str(centroid_pose)
                file1.write(str(self.most_recent_locations) + "\n")
                self.most_recent_locations[:,i] = centroid_pose
                if self.detect_time[:,i]==0:
                    t0 = time.time()
                    self.detect_time[:,i] = t0-start_time
                file2.write(str(self.detect_time) + "\n")
                print("tag", i, ": ", centroid_pose)

                if self.tag_estimates[i].queue[0] is None:
                    continue

                t = TransformStamped()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = 'map'

                t.child_frame_id = 'map_tag_' + str(i)
                t.transform.translation.x = centroid_pose[0]
                t.transform.translation.y = centroid_pose[1]
                t.transform.translation.z = centroid_pose[2]
                t.transform.rotation = self.tag_estimates[i].queue[0].orientation
                self.tf_broadcaster.sendTransform(t)

                marker = Marker()
                marker.header.frame_id = "map"
                marker.ns = "our_tag"
                marker.header.stamp = rospy.Time.now()
                marker.id = i
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.pose.position.x = centroid_pose[0]
                marker.pose.position.y = centroid_pose[1]
                marker.pose.position.z = centroid_pose[2]
                marker.scale.x = 0.20
                marker.scale.y = 0.20
                marker.scale.z = 0.20
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                self.vis_pub.publish(marker)


    def transform_pose(self, input_pose, init_frame, final_frame):
        # **Assuming /tf2 topic is being broadcasted
        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = input_pose
        pose_stamped.header.frame_id = init_frame
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = "camera_rgb_optical_frame"
        try:
            # ** It is important to wait for the listener to start listening. Hence the rospy.Duration(1)
            output_pose_stamped = self.tf_buffer.transform(pose_stamped, \
                final_frame, rospy.Duration(1))
            return output_pose_stamped.pose
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, \
            tf2_ros.ExtrapolationException):
            pass


if __name__ == "__main__":
    rospy.init_node('tag_updater')
    #rospy.init_node('odom_node')
    TagsUpdater()
    rospy.spin()
