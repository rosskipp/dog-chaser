import numpy as np

class VFH_ObstacleAvoidance:
    def __init__(self, lidar_range=270, angle_resolution=1, min_distance=0.3, max_distance=2.0):
        """
        Initialize VFH parameters.
        :param lidar_range: The range of the LiDAR scan in degrees (e.g., 270°).
        :param angle_resolution: The resolution of the LiDAR scan in degrees.
        :param min_distance: Minimum safe distance to an obstacle.
        :param max_distance: Maximum distance to consider obstacles.
        """
        self.lidar_range = lidar_range
        self.angle_resolution = angle_resolution
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_bins = int(lidar_range / angle_resolution)

    def process_lidar(self, lidar_data):
        """
        Process LiDAR data into a histogram.
        :param lidar_data: List of distance values from LiDAR.
        :return: Histogram of obstacle density.
        """
        # Normalize distances to range [0, 1]
        normalized_data = np.clip((self.max_distance - np.array(lidar_data)) / self.max_distance, 0, 1)

        # Identify obstacles (values closer to 1 indicate nearby objects)
        histogram = np.where(lidar_data < self.min_distance, 1, normalized_data)

        return histogram

    def find_clear_path(self, histogram):
        """
        Find the best path using histogram.
        :param histogram: Processed obstacle density data.
        :return: Best escape angle (in degrees).
        """
        # Threshold for detecting obstacles (0.5 means semi-blocked)
        obstacle_threshold = 0.5
        free_sectors = np.where(histogram < obstacle_threshold)[0]  # Indexes of free paths

        if len(free_sectors) == 0:
            return None  # No clear path found

        # Choose the widest gap (center of the largest cluster)
        gap_sizes = np.diff(np.concatenate(([-1], free_sectors, [self.num_bins])))
        max_gap_index = np.argmax(gap_sizes)
        best_sector = free_sectors[max_gap_index // 2]  # Midpoint of largest gap

        # Convert index to angle
        best_angle = best_sector * self.angle_resolution - (self.lidar_range / 2)
        return best_angle

    def compute_velocity(self, best_angle):
        """
        Compute linear and angular velocity based on escape direction.
        :param best_angle: Angle of the best path.
        :return: (linear_velocity, angular_velocity)
        """
        if best_angle is None:
            return 0.0, 0.5  # If no clear path, rotate in place

        # Scale speeds based on angle
        max_speed = 1.0
        max_turn = 1.0
        linear_velocity = max_speed * (1 - abs(best_angle) / 90)  # Reduce speed for sharp turns
        angular_velocity = max_turn * (best_angle / 90)  # Normalize to [-1,1]

        return linear_velocity, angular_velocity

    def run_vfh(self, lidar_data):
        """
        Main function to process LiDAR data and determine movement commands.
        :param lidar_data: Raw LiDAR scan data.
        :return: (linear_velocity, angular_velocity)
        """
        histogram = self.process_lidar(lidar_data)
        best_angle = self.find_clear_path(histogram)
        return self.compute_velocity(best_angle)