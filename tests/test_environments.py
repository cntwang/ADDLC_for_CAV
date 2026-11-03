"""Numerical and algorithm contracts for the shared road-width interface."""

import gc
import importlib.util
import unittest

import numpy as np

AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ["numba", "gymnasium", "torch", "stable_baselines3", "scipy"])


@unittest.skipUnless(AVAILABLE, "Install cav-dl[rl] for environment tests")
class RoadWidths(unittest.TestCase):
    def test_action_space_and_capacity_scaling(self):
        from cav_dl.envs.reward import make_env
        from cav_dl.envs._legacy import kernel as k
        for lanes in [3, 4, 5]:
            env = make_env(lanes)
            self.assertEqual(env.action_space.n, lanes ** 4)
            self.assertEqual(env.env.decode_discrete_action(lanes ** 4 - 1),
                             (lanes - 1,) * 4)
            state = np.zeros((120, len(k.FIELDS)))
            state[:, k.N] = 10 * lanes / 3
            state[:, k.P] = .5
            env.env.kernel.parameters_ulp(state)
            np.testing.assert_allclose(state[:, k.RHO], 10)
            expected = k.fd(.5)[1] * 15 / 3600 * lanes
            np.testing.assert_allclose(state[:, k.R], expected)
            env.close()
            del env
            gc.collect()

    def test_five_term_reward_identity(self):
        from cav_dl.envs.reward import make_env
        from cav_dl.data.synthetic import scene, MODES
        data = scene(710000000, .5, MODES[0], MODES[1])
        for lanes in [3, 4, 5]:
            env = make_env(lanes)
            obs, _ = env.reset(options=data, test=True)
            env.skip_reference_identity = True
            self.assertEqual(obs.shape, (38,))
            _, reward, _, _, info = env.step(0)
            self.assertAlmostEqual(reward, info["six_reward_components"][:5].sum(), places=9)
            env.close()
            del env
            gc.collect()

    def test_sac_supports_all_action_counts_and_terminal_flush(self):
        import torch
        from cav_dl.agents.sac import DiscreteSAC, ThreeStepReplay
        torch.set_num_threads(1)
        for actions in [81, 256, 625]:
            agent = DiscreteSAC(seed=17, actions=actions)
            batch = [torch.zeros((8, 38)), torch.arange(8) % actions,
                     torch.ones(8), torch.zeros((8, 38)), torch.zeros(8),
                     torch.ones(8) * .99**3]
            losses = agent.update(batch)
            self.assertTrue(np.isfinite(list(losses.values())).all())
            self.assertLess(agent.predict(np.zeros(38), True), actions)
        replay = ThreeStepReplay(capacity=10)
        replay.add(np.zeros(38), 0, 1., np.ones(38), False)
        replay.add(np.ones(38), 1, 2., np.ones(38) * 2, True)
        self.assertEqual(replay.size, 2)
        self.assertEqual(len(replay.pending), 0)
        np.testing.assert_allclose(replay.discount[:2], [.99**2, .99])

    def test_hv_cohort_is_retained_and_released_without_injection(self):
        from cav_dl.envs.stay_helpers import retained
        after, new, released = retained(0., 0., 1., 90., .5, .2)
        self.assertAlmostEqual(after, 3.)
        self.assertAlmostEqual(new, 3.)
        self.assertAlmostEqual(released, 0.)
        after, new, released = retained(after, 1., 0., 90., .5, .2)
        self.assertAlmostEqual(after, 0.)
        self.assertAlmostEqual(new, 0.)
        self.assertAlmostEqual(released, 3.)
