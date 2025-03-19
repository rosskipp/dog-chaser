import unittest
import numpy as np
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from kalman import Kalman3DAcceleration

class TestKalman3DAcceleration(unittest.TestCase):
    def setUp(self):
        """Set up a new Kalman filter instance before each test"""
        self.kalman = Kalman3DAcceleration(max_missed_frames=3)

    def test_initialization(self):
        """Test if Kalman filter is initialized with correct values"""
        # Check initial tracking status
        self.assertFalse(self.kalman.get_tracking())

        # Check initial state
        initial_state = self.kalman.get_state()
        self.assertEqual(initial_state.shape, (9, 1))
        np.testing.assert_array_equal(initial_state, np.zeros((9, 1), dtype=np.float32))

    def test_prediction_without_measurement(self):
        """Test prediction without any prior measurements"""
        prediction = self.kalman.predict()
        self.assertEqual(prediction.shape, (9, 1))
        # First prediction should still be zeros since no measurements yet
        np.testing.assert_array_equal(prediction, np.zeros((9, 1), dtype=np.float32))

    def test_correction_with_measurement(self):
        """Test correction with a valid measurement"""
        measurement = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.kalman.predict()
        self.kalman.correct(measurement)

        # Check if tracking is enabled after measurement
        self.assertTrue(self.kalman.get_tracking())

        # Position should be close to measurement (not exact due to Kalman filtering)
        position = self.kalman.get_position()
        self.assertEqual(position.shape, (3, 1))
        np.testing.assert_array_almost_equal(
            position.flatten(), measurement, decimal=1
        )

    def test_tracking_loss(self):
        """Test if tracking is lost after max_missed_frames"""
        # First give it a measurement to start tracking
        measurement = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.kalman.predict()
        self.kalman.correct(measurement)
        self.assertTrue(self.kalman.get_tracking())

        # Miss frames until tracking is lost
        for _ in range(3):  # max_missed_frames is 3
            self.kalman.predict()
            self.kalman.correct(None)

        # Check if tracking is lost
        self.assertFalse(self.kalman.get_tracking())

        # Check if state is reset
        np.testing.assert_array_equal(
            self.kalman.get_state(),
            np.zeros((9, 1), dtype=np.float32)
        )

    def test_velocity_estimation(self):
        """Test velocity estimation with consecutive measurements"""
        # First measurement
        m1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.kalman.predict()
        self.kalman.correct(m1)

        # Second measurement with displacement
        m2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.kalman.predict()
        self.kalman.correct(m2)

        # Get velocity estimate
        velocity = self.kalman.get_velocity()
        self.assertEqual(velocity.shape, (3, 1))
        # Velocity should be positive in all dimensions
        self.assertTrue(np.all(velocity > 0))

    def test_acceleration_estimation(self):
        """Test acceleration estimation with consecutive measurements"""
        measurements = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
            np.array([4.0, 4.0, 4.0], dtype=np.float32)
        ]

        for m in measurements:
            self.kalman.predict()
            self.kalman.correct(m)

        # Get acceleration estimate
        acceleration = self.kalman.get_acceleration()
        self.assertEqual(acceleration.shape, (3, 1))
        # Acceleration should be positive in all dimensions
        self.assertTrue(np.all(acceleration > 0))

if __name__ == '__main__':
    unittest.main()