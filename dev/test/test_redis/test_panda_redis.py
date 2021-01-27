"""Test script for panda specific redis interface.
"""

import pytest
import redis
import numpy as np
from perls2.redis_interfaces.redis_interface import PandaRedisInterface
from perls2.redis_interfaces.redis_keys import *
from perls2.redis_interfaces.redis_values import *
import perls2.redis_interfaces.panda_redis_keys as P
from dev.test.test_panda.fake_franka_panda import FakePandaDriver



@pytest.fixture
def panda_redis():
	panda_redis = PandaRedisInterface(host='127.0.0.1', port=6379)
	return panda_redis

@pytest.fixture
def fake_driver():
	fake_driver = FakePandaDriver()
	fake_driver.set_fake_state()
	return fake_driver

def test_env_connected(panda_redis):
	 panda_redis._client.set(ROBOT_ENV_CONN_KEY, 'True')
	 assert(panda_redis.is_env_connected())

	 # test other values than true.
	 panda_redis._client.set(ROBOT_ENV_CONN_KEY, 'NOT TRUE')
	 assert(panda_redis.is_env_connected() == False)

def test_get_driver_states(panda_redis, fake_driver):
	driver_states = panda_redis.get_driver_states()
	for key, state in driver_states.items():
		if key == P.ROBOT_STATE_EE_POSE_KEY:
			ee_pose = driver_states[P.ROBOT_STATE_EE_POSE_KEY]
			assert(ee_pose.shape == P.EE_POSE_SHAPE)
			# make sure properly imported (column-major).
			# last row should be 0, 0, 0, 1 (homogeneous transform)
			assert np.all(ee_pose[3, :] == [0, 0, 0, 1])
			# make sure rotation matrix is orthogonal.
			rot_m =  ee_pose[0:3, 0:3]
			print(np.dot(rot_m, rot_m.T))
			assert np.all(np.isclose(np.dot(rot_m, rot_m.T), np.eye(3), rtol=1e-5, atol=1e-5))
			# check values for ee_position (last column, 1st 3 rows.)
			ee_pos = ee_pose[0:3, 3]
			assert np.all(np.isclose(ee_pos,[0.534449340443935, 0.0364261218566121, 0.313053490964604]))


		else:
			assert(driver_states[key].shape == (7,))

def test_get_driver_state_model(panda_redis, fake_driver):
	driver_states = panda_redis.get_driver_state_model()
	# check all keys present.
	state_model_keys = P.ROBOT_STATE_KEYS + P.ROBOT_MODEL_KEYS
	key_present = [key in driver_states.keys() for key in state_model_keys]
	assert (np.all(key_present))

	for key, state in driver_states.items():
		if key == P.ROBOT_STATE_EE_POSE_KEY:
			ee_pose = driver_states[P.ROBOT_STATE_EE_POSE_KEY]
			assert(ee_pose.shape == P.EE_POSE_SHAPE)
			# make sure properly imported (column-major).
			# last row should be 0, 0, 0, 1 (homogeneous transform)
			assert np.all(ee_pose[3, :] == [0, 0, 0, 1])
			# make sure rotation matrix is orthogonal.
			rot_m =  ee_pose[0:3, 0:3]
			print(np.dot(rot_m, rot_m.T))
			assert np.all(np.isclose(np.dot(rot_m, rot_m.T), np.eye(3), rtol=1e-5, atol=1e-5))
			# check values for ee_position (last column, 1st 3 rows.)
			ee_pos = ee_pose[0:3, 3]
			assert np.all(np.isclose(ee_pos,[0.534449340443935, 0.0364261218566121, 0.313053490964604]))
		elif key == P.ROBOT_MODEL_JACOBIAN_KEY:
			jacobian = driver_states[P.ROBOT_MODEL_JACOBIAN_KEY]
			assert (jacobian.shape == P.JACOBIAN_SHAPE)
		elif key == P.ROBOT_MODEL_MASS_MATRIX_KEY:
			mass_matrix = driver_states[P.ROBOT_MODEL_MASS_MATRIX_KEY]
			assert(mass_matrix.shape == P.MASS_MATRIX_SHAPE)

		else:
			assert(driver_states[key].shape == (7,))

