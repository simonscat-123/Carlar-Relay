import unittest

from carla_relay.experiments.localization.fusion import (
    AxisKalmanFilter,
    PositionKalmanFilter2D,
    SensorFrameGate,
    YawKalmanFilter,
    velocity_increment_std,
)


class LocalizationFusionTests(unittest.TestCase):
    def test_velocity_random_walk_discretization_is_rate_independent(self):
        sigma = 0.2
        total_seconds = 10.0
        var_20_hz = sum(velocity_increment_std(sigma, 0.05) ** 2 for _ in range(200))
        var_100_hz = sum(velocity_increment_std(sigma, 0.01) ** 2 for _ in range(1000))
        expected = sigma**2 * total_seconds
        self.assertAlmostEqual(var_20_hz, expected)
        self.assertAlmostEqual(var_100_hz, expected)

    def test_cached_gnss_frame_is_only_accepted_once(self):
        gate = SensorFrameGate()
        self.assertTrue(gate.accept(100))
        for _ in range(20):
            self.assertFalse(gate.accept(100))
        self.assertTrue(gate.accept(120))

    def test_position_prediction_uses_acceleration(self):
        filt = AxisKalmanFilter(position=0.0, velocity=2.0,
                                position_variance=0.0, velocity_variance=0.0)
        filt.predict(acceleration=1.0, dt=2.0, velocity_random_walk_std=0.0)
        self.assertAlmostEqual(filt.position, 6.0)
        self.assertAlmostEqual(filt.velocity, 4.0)

    def test_artificial_ins_error_enters_kalman_velocity_state(self):
        filt = AxisKalmanFilter(position_variance=0.0, velocity_variance=0.0)
        filt.predict(0.0, 0.05, 0.2, velocity_increment_error=0.3)
        self.assertAlmostEqual(filt.position, 0.0)
        self.assertAlmostEqual(filt.velocity, 0.3)
        filt.predict(0.0, 0.05, 0.2, velocity_increment_error=0.0)
        self.assertAlmostEqual(filt.position, 0.015)

    def test_noisier_gnss_has_lower_kalman_gain(self):
        low_noise = AxisKalmanFilter(position_variance=4.0, velocity_variance=1.0)
        high_noise = AxisKalmanFilter(position_variance=4.0, velocity_variance=1.0)
        low_gain, _, _ = low_noise.update(10.0, gnss_position_std=0.5)
        high_gain, _, _ = high_noise.update(10.0, gnss_position_std=4.0)
        self.assertGreater(low_gain, high_gain)

    def test_process_uncertainty_increases_next_gnss_gain(self):
        early = AxisKalmanFilter(position_variance=0.1, velocity_variance=0.0)
        late = AxisKalmanFilter(position_variance=0.1, velocity_variance=0.0)
        for _ in range(100):
            late.predict(0.0, 0.05, 0.4)
        early_gain, _, _ = early.update(1.0, 1.0)
        late_gain, _, _ = late.update(1.0, 1.0)
        self.assertGreater(late_gain, early_gain)

    def test_position_update_corrects_position_and_velocity(self):
        filt = PositionKalmanFilter2D(initial_position_std=2.0, initial_velocity_std=1.0)
        filt.predict(0.0, 0.0, 1.0, 0.1)
        diag = filt.update(10.0, -5.0, gnss_position_std=0.5)
        self.assertGreater(filt.x.position, 0.0)
        self.assertLess(filt.y.position, 0.0)
        self.assertGreater(diag["gain_position_x"], 0.0)
        self.assertGreater(diag["gain_velocity_x"], 0.0)

    def test_yaw_filter_wraps_innovation_across_180_degrees(self):
        filt = YawKalmanFilter(179.0, initial_std_deg=5.0)
        filt.predict(0.0, 0.05, 1.0)
        gain, innovation = filt.update(-179.0, measurement_std_deg=1.0)
        self.assertAlmostEqual(innovation, 2.0)
        self.assertGreater(gain, 0.0)
        self.assertLess(abs(abs(filt.angle_deg) - 180.0), 2.0)


if __name__ == "__main__":
    unittest.main()
