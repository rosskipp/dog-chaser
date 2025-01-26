import numpy as np
import cv2


class Kalman3DAcceleration:
    def __init__(self, max_missed_frames=10):
        self.tracking = False
        self.kf = cv2.KalmanFilter(
            9, 3
        )  # 9 states (x, y, z, vx, vy, vz, ax, ay, az), 3 measurements (x, y, z)
        self.max_missed_frames = max_missed_frames
        self.consecutive_missed_frames = 0  # Tracks consecutive misses

        dt = 1  # Time step
        dt2 = 0.5 * dt**2  # Acceleration factor

        # State Transition Matrix (A)
        self.kf.transitionMatrix = np.array(
            [
                [1, 0, 0, dt, 0, 0, dt2, 0, 0],
                [0, 1, 0, 0, dt, 0, 0, dt2, 0],
                [0, 0, 1, 0, 0, dt, 0, 0, dt2],
                [0, 0, 0, 1, 0, 0, dt, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, dt, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, dt],
                [0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Measurement Matrix (H)
        self.kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )

        # Process Noise Covariance (Q)
        self.kf.processNoiseCov = np.eye(9, dtype=np.float32) * 0.03

        # Measurement Noise Covariance (R)
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.05

        # Error Covariance Matrix (P)
        self.kf.errorCovPost = np.eye(9, dtype=np.float32) * 1

        # Initialize State
        self.reset_state()

    def reset_state(self) -> None:
        """Resets the Kalman filter state and consecutive miss counter"""
        self.kf.statePost = np.zeros((9, 1), dtype=np.float32)
        self.consecutive_missed_frames = 0

    def predict(self) -> np.ndarray:
        """Predicts the next state based on previous state"""
        return self.kf.predict()

    def correct(self, measurement: np.ndarray | None) -> None:
        """Updates the Kalman filter with a new measurement"""
        if measurement is not None:
            self.tracking = True
            self.consecutive_missed_frames = 0  # Reset consecutive miss count
            return self.kf.correct(measurement)
        else:
            # we recieve no measurement, so we increment the missed frames counter
            self.consecutive_missed_frames += 1
            if self.consecutive_missed_frames >= self.max_missed_frames:
                print(
                    "\n[INFO] Object lost after consecutive misses! Resetting Kalman filter."
                )
                self.reset_state()
                self.tracking = False
            return None  # No correction applied

    def get_tracking(self) -> bool:
        """Returns the current tracking status"""
        return self.tracking

    def get_state(self) -> np.ndarray:
        """Returns the current estimated position, velocity, and acceleration"""
        return self.kf.statePost

    def get_position(self) -> np.ndarray:
        """Returns the current estimated position"""
        return self.kf.statePost[:3]

    def get_velocity(self) -> np.ndarray:
        """Returns the current estimated velocity"""
        return self.kf.statePost[3:6]

    def get_acceleration(self) -> np.ndarray:
        """Returns the current estimated acceleration"""
        return self.kf.statePost[6:9]
