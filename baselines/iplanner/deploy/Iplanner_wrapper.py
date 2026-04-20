import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from iplanner_agent import IPlannerAgent
from tracking_utils import MPC_Controller
from my_interfaces.msg import DepthGoal # Tem que colocar a mensagem nova aqui
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import numpy as np
import threading

# TODO: Intrinsic e args

class PlanningOutput:
    def __init__(self):
        self.trajectory_points_world = None
        self.all_trajectories_world = None
        self.all_values_camera = None

class PlannerNode(Node):

    def __init__(self):
        super().__init__('planner_node')

        self.declare_parameter('intrinsic', [0.0, 0.0])
        self.declare_parameter('checkpoint', [0.0])
        self.declare_parameter('config', [0.0])
        self.declare_parameter('speed', 0.0)
        self.declare_parameter('goal_range', 0.0)
        self.declare_parameter('controller_frequency', 30.0)

        intrinsic = np.array(self.get_parameter('intrinsic').get_parameter_value().double_value)
        checkpoint = np.array(self.get_parameter('checkpoint').get_parameter_value().string_value)
        config = np.array(self.get_parameter('config').get_parameter_value().string_value)

        self.planner = IPlannerAgent(intrinsic,
                                    model_path=checkpoint,
                                    model_config_path=config,
                                    device='cuda:0')

        self.bridge = CvBridge()
        self.mutex_planner = threading.Lock() # Planner trajectory mutex
        self.mutex_odom = threading.Lock() # Odometry value mutex
        self.freq_controller = self.get_parameter('controller_frequency').get_parameter_value().double_value # Maximum controller frequency in hz
        self.planner_group = MutuallyExclusiveCallbackGroup()
        self.odometry_group = MutuallyExclusiveCallbackGroup()

        self.planning_output = PlanningOutput()
        self.mpc = None
        self.current_odom = None
        self.goal_range = self.get_parameter('goal_range').get_parameter_value().double_value
        self.speed = self.get_parameter('speed').get_parameter_value().double_value

        self.planner_sub = self.create_subscription(
            DepthGoal,
            '/iplanner_input',
            self.planning,
            1,
            callback_group=self.planner_group
        )

        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/iplanner_output',
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.update_odometry,
            1,
            callback_group=self.odometry_group
        )

        self.control_loop = self.create_timer(1.0 / self.freq_controller, self.controller)

        self.get_logger().info("Planner node started.")

    def planning(self, msg: DepthGoal):
        try:
            # 🔹 goal
            goal_x = np.array(msg.goal.x)
            goal_y = np.array(msg.goal.y)
            goal = np.stack((goal_x,goal_y,np.zeros_like(goal_x)),axis=1)
            goal = np.array([msg.goal.x, msg.goal.y], dtype=np.float32)
            goal = np.clip(goal, -self.goal_range, self.goal_range)
            batch_size = goal.shape[0]

            # 🔹 depth
            depth = self.bridge.imgmsg_to_cv2(
                msg.depth, desired_encoding='passthrough'
            )
            depth = depth.astype(np.float32) / 10000.0
            depth = depth.reshape((batch_size, -1, depth.shape[1], 1))

            # 🔹 planner_output
            _, trajectory, fear = self.planner.step_pointgoal(depth, goal)

            trajectory_points_camera = trajectory.cpu().numpy()
            all_trajectories_camera = trajectory.cpu().numpy()[None, :, :, :]
            all_values_camera = fear.cpu().numpy()

            with self.mutex_odom:
                odom = self.current_odom
            # Transform trajectory from camera frame to world frame
            trajectory_points_camera_single = trajectory_points_camera[0]            
            trajectory_points_world = []
            for point in trajectory_points_camera_single:
                point_local = np.array([point[0], point[1], 0.0])
                point_world = camera_pos[0] + camera_rot[0] @ point_local
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
                    point_world = camera_pos[0] + camera_rot[0] @ point_local
                    traj_world.append(point_world[:2])
                all_trajectories_world.append(np.array(traj_world))

            # batch format (1, N_traj, T, 2)
            batch_all_points_world = np.array(all_trajectories_world)[None, ...]
            
            new_planning_output = PlanningOutput()

            new_planning_output.trajectory_points_world = batch_optimal_points_world
            new_planning_output.all_trajectories_world = batch_all_points_world
            new_planning_output.all_values_camera = all_values_camera

            with self.mutex_planner:
                self.planning_output = new_planning_output
                self.mpc = mpc

        except Exception as e:
            self.get_logger().error(f"Erro: {e}")

    
    def controller(self):
        x0 = np.stack([camera_pos[:,0], camera_pos[:,1], np.arctan2(camera_rot[:,1,0], camera_rot[:,0,0]), [robot_vel], [robot_ang_vel]],axis=-1)
        
        with self.mutex_planner:
            current_planning = self.planning_output
            mpc = self.mpc

        if current_planning.trajectory_points_world is not None:
            current_trajectory = current_planning.trajectory_points_world if current_planning.trajectory_points_world is not None else None
            current_all_trajectories = current_planning.all_trajectories_world if current_planning.all_trajectories_world is not None else None
            current_all_values = current_planning.all_values_camera if current_planning.all_values_camera is not None else None
        
        if mpc is None:
            return

        opt_u_controls, opt_x_states = mpc.solve(x0[:3])
        v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
    
    def update_odometry(self, odom):
        with self.mutex_odom:
            self.current_odom = odom


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()