import unittest

from ko_estimation import KoopmanEstimator
from simulations import OUDeterministic


class MyTestCase(unittest.TestCase):

    def test_one_d_estimation(self):
        n = 1000
        x = OUDeterministic(n).reshape(-1, 1)

        koopman_estimator = KoopmanEstimator(x)
        koopman_estimator.estimate_density(epsilon_0=0.2)

        self.assertAlmostEqual(koopman_estimator.p_est(0)[0], 0.39166897, places=5)
        self.assertAlmostEqual(koopman_estimator.p_est(1)[0], 0.24212, places=5)
        self.assertAlmostEqual(koopman_estimator.p_est(-1)[0], 0.24212, places=5)


if __name__ == "__main__":
    unittest.main()
