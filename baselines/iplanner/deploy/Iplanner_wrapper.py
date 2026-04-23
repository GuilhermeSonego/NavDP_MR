import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from iplanner_agent import IPlannerAgent
from tracking_utils import MPC_Controller
from visualization_utils import VisualizationManager
from basic_utils import draw_box_with_text

from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

import numpy as np
import threading
import imageio
import cv2

# TODO: Intrinsic e args

class PlanningOutput:
    def __init__(self):
        self.trajectory_points_world = None
        self.all_trajectories_world = None
        self.all_values_camera = None

class PlannerNode(Node):

    def __init__(self):
        super().__init__('planner_node')

        self.declare_parameter('intrinsic', [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter('checkpoint', '')
        self.declare_parameter('config', '')
        self.declare_parameter('speed', 0.0)
        self.declare_parameter('goal_range', 0.0)
        self.declare_parameter('controller_max_frequency', 60.0)
        self.declare_parameter('planner_max_frequency', 60.0)
        self.declare_parameter('depth_height', 0)
        self.declare_parameter('depth_width', 0)

        self.depth_height = self.get_parameter('depth_height').get_parameter_value().integer_value
        self.depth_width = self.get_parameter('depth_width').get_parameter_value().integer_value

        self.intrinsic = np.array(self.get_parameter('intrinsic').get_parameter_value().double_array_value)
        checkpoint = self.get_parameter('checkpoint').get_parameter_value().string_value
        config = self.get_parameter('config').get_parameter_value().string_value

        self.planner = IPlannerAgent(self.intrinsic,
                                    model_path=checkpoint,
                                    model_config_path=config,
                                    device='cuda:0')

        self.mutex_planner = threading.Lock() # Planner trajectory mutex
        self.mutex_odom = threading.Lock() # Odometry value mutex
        self.mutex_pointcloud = threading.Lock() # Pointcloud mutex
        self.mutex_goal = threading.Lock() # Current goal mutex
        self.mutex_image = threading.Lock() # Last image mutex
        self.freq_controller = self.get_parameter('controller_max_frequency').get_parameter_value().double_value # Maximum controller frequency in hz
        self.freq_planner = self.get_parameter('planner_max_frequency').get_parameter_value().double_value # Maximum planner frequency in hz
        self.planner_group = MutuallyExclusiveCallbackGroup()
        self.controller_group = MutuallyExclusiveCallbackGroup()
        self.odometry_group = MutuallyExclusiveCallbackGroup()
        self.pointcloud_group = MutuallyExclusiveCallbackGroup()
        self.goal_group = MutuallyExclusiveCallbackGroup()
        self.image_group = MutuallyExclusiveCallbackGroup()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cv_bridge = CvBridge()
        self.planning_output = PlanningOutput()
        self.mpc = None
        self.current_goal_world = None
        self.current_odom = None
        self.current_pointcloud = None
        self.current_image = None
        self.vis_depth = None
        self.vis_image = None
        self.vis_goal = None
        self.goal_range = self.get_parameter('goal_range').get_parameter_value().double_value
        self.speed = self.get_parameter('speed').get_parameter_value().double_value


        save_dir = './trajectories/'
        self.fps_writer = imageio.get_writer(save_dir + "fps.mp4", fps=10)
        self.vis_manager = VisualizationManager(history_size=5)

        self.odom_sub = self.create_subscription(
            Odometry,
            '/local_position/odom',
            self.update_odometry,
            1,
            callback_group=self.odometry_group
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/pointcloud',
            self.update_pointcloud,
            1,
            callback_group = self.pointcloud_group
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.update_image,
            1,
            callback_group=self.image_group
        )

        self.goal_sub = self.create_subscription(
            PointStamped,
            '/waypoint',
            self.update_goal,
            1,
            callback_group=self.goal_group
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.planner_loop = self.create_timer(
            1.0 / self.freq_planner,
            self.planning,
            callback_group=self.planner_group  
        )

        self.control_loop = self.create_timer(
            1.0 / self.freq_controller,
            self.controller,
            callback_group=self.controller_group
        )

        self.get_logger().info("Planner node started.")

    def planning(self):
        try:
            with self.mutex_goal:
                goal_world = self.current_goal_world

            if goal_world is None:
                self.get_logger().warn('(Planner): No goal available.')
                return

            with self.mutex_odom:
                current_odom = self.current_odom

            if current_odom is None:
                self.get_logger().warn('(Planner): No odometry detected for planning.')
                return

            # transforms goal in world frame to relative to camera frame
            pos = current_odom.pose.pose.position
            quat = current_odom.pose.pose.orientation
            camera_rot = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
            camera_pos = np.array([pos.x, pos.y, pos.z])

            goal_relative = camera_rot.T @ (np.array([goal_world.x, goal_world.y, goal_world.z]) - camera_pos)

            goal_x = np.array([goal_relative[0]], dtype=np.float32)
            goal_y = np.array([goal_relative[1]], dtype=np.float32)
            goal = np.stack((goal_x, goal_y, np.zeros_like(goal_x)), axis=1)
            goal = np.clip(goal, -self.goal_range, self.goal_range)
            batch_size = goal.shape[0]

            with self.mutex_pointcloud:
                pointcloud = self.current_pointcloud
             
            if pointcloud is None:
                self.get_logger().warn('(Planner): No pointcloud detected for planning.')
                return

            depth = self.pointcloud2_to_depth(pointcloud, self.depth_width, self.depth_height)
            depth = depth.astype(np.float32)
            depth_for_vis = depth.copy()
            depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

            # 🔹 planner_output
            _, trajectory, fear = self.planner.step_pointgoal(depth, goal)

            trajectory_points_camera = trajectory.cpu().numpy()
            all_trajectories_camera = trajectory.cpu().numpy()[None, :, :, :]
            all_values_camera = fear.cpu().numpy()

            # Transform trajectory from camera frame to world frame
            trajectory_points_camera_single = trajectory_points_camera[0]            
            trajectory_points_world = []
            for point in trajectory_points_camera_single:
                point_local = np.array([point[0], point[1], 0.0])
                point_world = camera_pos + camera_rot @ point_local
                trajectory_points_world.append(point_world[:2])
            trajectory_points_world = np.array(trajectory_points_world)
            batch_optimal_points_world = trajectory_points_world[None, :, :]  # shape (1, T, 2)
            
            # initialize controller based on 
            mpc = MPC_Controller(trajectory_points_world,
                                    desired_v=self.speed,
                                    v_max=self.speed,
                                    w_max=self.speed)

            all_trajectories_camera_single = all_trajectories_camera[0]

            all_trajectories_world = []
            for traj_camera in all_trajectories_camera_single:
                traj_world = []
                for point in traj_camera:
                    point_local = np.array([point[0], point[1], 0.0])
                    point_world = camera_pos + camera_rot @ point_local
                    traj_world.append(point_world[:2])
                all_trajectories_world.append(np.array(traj_world))

            # batch format (1, N_traj, T, 2)
            batch_all_points_world = np.array(all_trajectories_world)[None, ...]
            
            new_planning_output = PlanningOutput()

            new_planning_output.trajectory_points_world = batch_optimal_points_world
            new_planning_output.all_trajectories_world = batch_all_points_world
            new_planning_output.all_values_camera = all_values_camera

            with self.mutex_image:
                current_image = self.current_image

            with self.mutex_planner:
                self.planning_output = new_planning_output
                self.mpc = mpc
                self.vis_depth = depth_for_vis
                self.vis_image = current_image
                self.vis_goal = goal_relative

        except Exception as e:
            self.get_logger().error(f"Erro: {e}")

    
    def controller(self):
        with self.mutex_odom:
            current_odom = self.current_odom
        
        if current_odom is None:
            self.get_logger().warn('(Controller): No odometry detected for control.')
            return
        # position
        pos = current_odom.pose.pose.position
        camera_pos = np.array([pos.x, pos.y, pos.z])

        # orientation
        quat = current_odom.pose.pose.orientation
        camera_rot = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

        # velocity
        robot_vel = current_odom.twist.twist.linear.x
        robot_ang_vel = current_odom.twist.twist.angular.z
        
        x0 = np.array([camera_pos[0], camera_pos[1], np.arctan2(camera_rot[1,0], camera_rot[0,0]), robot_vel, robot_ang_vel])
        
        with self.mutex_planner:
            current_planning = self.planning_output
            mpc = self.mpc
            images = self.vis_image
            depths = self.vis_depth
            goals = self.vis_goal
        
        if mpc is None:
            self.get_logger().warn('(Controller): No MPC detected for control.')
            return

        opt_u_controls, opt_x_states = mpc.solve(x0[:3])
        v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]

        theta = opt_x_states[1, 2]  
        cmd_vel = Twist()
        cmd_vel.linear.x = float(v * np.cos(theta))
        cmd_vel.linear.y = float(v * np.sin(theta))
        cmd_vel.angular.z = float(w)

        self.cmd_vel_pub.publish(cmd_vel)

        current_trajectory = current_planning.trajectory_points_world
        current_all_trajectories = current_planning.all_trajectories_world
        current_all_values = current_planning.all_values_camera

        if images is None or depths is None or goals is None or current_trajectory is None:
            return

        try:
            vis_image = self.vis_manager.visualize_trajectory(
                images, depths[:,:,None], self.intrinsic,
                current_trajectory,
                robot_pose=x0,
                all_trajectories_points=current_all_trajectories,
                all_trajectories_values=current_all_values
            )
            # Visualization
            vis_image = draw_box_with_text(vis_image,0,0,430,50,"desired lin.:%.2f ang.:%.2f"%(v,w))
            vis_image = draw_box_with_text(vis_image,0,50,430,50,"actual lin.:%.2f ang.:%.2f"%(robot_vel,robot_ang_vel))
            if current_all_values is not None:
                vis_image = draw_box_with_text(vis_image,0,770,430,50,"critic max:%.2f min:%.2f"%(np.max(current_all_values), np.min(current_all_values)))
            vis_image = draw_box_with_text(vis_image,0,820,430,50,"point goal:(%.2f, %.2f)"%(goals[0],goals[1]))
            cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            self.fps_writer.append_data(vis_image)
        except Exception as e:
            self.get_logger().warn(f'(Controller): Visualization failed: {e}')

    def update_odometry(self, odom):
        with self.mutex_odom:
            self.current_odom = odom

    def update_image(self, msg: Image):
        with self.mutex_image:
            self.current_image = self.cv_bridge.imgmsg_to_cv2(
                msg, desired_encoding='rgb8'
        )

    def update_pointcloud(self, pointcloud: PointCloud2):
        with self.mutex_pointcloud:
            self.current_pointcloud = pointcloud 

    def update_goal(self, msg: PointStamped):
        try:
            # transforms goal to world frame 
            transform = self.tf_buffer.lookup_transform(
                'world',
                msg.header.frame_id,
                rclpy.time.Time()
            )
            goal_world = do_transform_point(msg, transform)
            with self.mutex_goal:
                self.current_goal_world = goal_world.point  
        except Exception as e:
            self.get_logger().warn(f'(Goal): TF transform failed: {e}')

    def pointcloud2_to_depth(self, msg: PointCloud2, width: int, height: int) -> np.ndarray:
        intrinsic = self.intrinsic  # sua matriz intrínseca

        points_raw = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        points = np.array(list(points_raw), dtype=np.float32)

        if points.shape[0] == 0:
            return np.zeros((height, width), dtype=np.float32)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        valid = points[:, 2] > 0
        points = points[valid]

        u = (fx * points[:, 0] / points[:, 2] + cx).astype(int)
        v = (fy * points[:, 1] / points[:, 2] + cy).astype(int)

        mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z = u[mask], v[mask], points[mask, 2]

        depth_image = np.full((height, width), np.inf, dtype=np.float32)
        idx = np.argsort(z)
        depth_image[v[idx], u[idx]] = z[idx]
        depth_image[depth_image == np.inf] = 0.0

        return depth_image

def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()