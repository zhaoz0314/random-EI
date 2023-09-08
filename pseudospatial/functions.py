import jax # for jax.jit
import jax.numpy as jnp
import jax.random as jrandom
import time
# # allow double floats
# import jax.config as jconfig
# jconfig.update("jax_enable_x64", True)
# # to check
# jrandom.uniform(jrandom.PRNGKey(0), (1000, ), dtype=jnp.float64).dtype



# python: variable_name, ClassName, GLOBAL_VARIABLE_NAME
# MMA: variableName, selfFunction, $globalVariableName



# for obtaining conditions
# sub population sizes
def sub_part_n_s_fct(part_n, sub_part_r_s):
  # calculate rough numbers
  sub_part_n_s = jnp.floor(part_n * sub_part_r_s)
  # in case the rounded numbers don't add up
  sub_part_n_s = sub_part_n_s.at[-1].set(part_n - jnp.sum(sub_part_n_s[:-1]))
  return(sub_part_n_s.astype(int))
# sub_part_n_s_fct = jax.jit(sub_part_n_s_fct)

# (scaled or unscaled) mean calculator
def ei_mean_balancer(ei_part_n_s, e_mean):
  return(jnp.tile(jnp.asarray([e_mean, - e_mean * ei_part_n_s[0] / ei_part_n_s[1]]), (2, 1)))

# connectivities
def connectivity_s_generator(sub_part_n_s, unscaled_mean, unscaled_std,
                             connectivity_n,
                             key = jnp.array([0, 0], dtype = jnp.uint32)):
  # find particle number and arrangement of blocks
  part_n = jnp.sum(sub_part_n_s)
  block_shape = unscaled_mean.shape
  block_n = unscaled_mean.size
  # split key
  [key, *subkey_s] = jrandom.split(key, block_n + 1)
  # first for each block position generate for all connectivities
  stacked_block_s = [[unscaled_mean[row_idx, column_idx] / jnp.sqrt(part_n)
                      + jnp.multiply(
                        unscaled_std[row_idx, column_idx] / jnp.sqrt(part_n),
                        jrandom.normal(
                          subkey_s[row_idx * block_shape[0] + column_idx],
                          (connectivity_n, sub_part_n_s[row_idx], sub_part_n_s[column_idx])))
                      for column_idx in range(block_shape[1])]
                     for row_idx in range(block_shape[0])]
  # then build connectivities using blocks
  return([jnp.block(stacked_block_s), key])

#sparsify then row sum removal
def row_sum_removing_sparsifier(keyed_connectivity_s,
                                sub_part_n_s, in_part_r, rewiring_prob):
  # find different variables
  [connectivity_s, key] = keyed_connectivity_s
  connectivity_n = connectivity_s.shape[0]
  part_n = jnp.sum(sub_part_n_s)
  sub_pop_n = sub_part_n_s.shape[0]
  # generate adjacencies for nearby and faraway separately
  [key, *subkey_s] = jrandom.split(key, sub_pop_n * 2 + 1)
  nearby_prob = 1 - rewiring_prob * (1 - in_part_r)
  faraway_prob = rewiring_prob * in_part_r
  closeness_s = jnp.ceil(sub_part_n_s * in_part_r / 2 - 1).astype(int)
  adjacency_segment_s = [
    jnp.concatenate([jrandom.bernoulli(subkey_s[sub_pop_idx * 2 + 0], nearby_prob,
                                       (connectivity_n,
                                        part_n,
                                        2 * closeness_s[sub_pop_idx] + 1)),
                     jrandom.bernoulli(subkey_s[sub_pop_idx * 2 + 1], faraway_prob,
                                       (connectivity_n,
                                        part_n,
                                        sub_part_n_s[sub_pop_idx]
                                        - 2 * closeness_s[sub_pop_idx] - 1))],
                    axis = -1, dtype = bool)
    for sub_pop_idx in range(sub_pop_n)]
  # roll then concatenate segments
  sub_part_r_s = sub_part_n_s / part_n
  for sub_pop_idx in range(sub_pop_n):
    for row_idx in range(part_n):
      adjacency_segment_s[sub_pop_idx] = adjacency_segment_s[sub_pop_idx].at[
        :, row_idx].set(jnp.roll(adjacency_segment_s[sub_pop_idx][:, row_idx],
                               jnp.floor(row_idx * sub_part_r_s[sub_pop_idx])
                               - closeness_s[sub_pop_idx],
                               axis = -1))
  adjacency_s = jnp.concatenate(adjacency_segment_s, axis = -1, dtype = bool)
  # sparsify and rescale by in_part_r * part_n
  connectivity_s = connectivity_s * adjacency_s.astype(int) / jnp.sqrt(in_part_r)
  # only remove row mean (ignoring zeros) for nonzero entries
  connectivity_s = (connectivity_s
                    - adjacency_s.astype(int) * jnp.mean(
                      connectivity_s, axis = -1, where = adjacency_s,
                      keepdims = True))
  return([connectivity_s, key])
row_sum_removing_sparsifier = jax.jit(row_sum_removing_sparsifier)

# prediction of spectral radius
def ei_spectral_radius_fct(e_part_r, ei_std_s):
  return(jnp.sqrt(ei_std_s[0] ** 2 * e_part_r
                  + ei_std_s[1] ** 2 * (1 - e_part_r)))

# external connectivities
def ext_connectivity_s_generator(sub_part_n_s, ext_sub_part_n_s,
                                 unscaled_ext_mean, unscaled_ext_std,
                                 ext_connectivity_n, 
                                 key = jnp.array([0, 0], dtype = jnp.uint32)):
  # find number of sub populations
  sub_pop_n = unscaled_ext_mean.shape[0]
  ext_sub_pop_n = unscaled_ext_mean.shape[1]
  ext_part_n = jnp.sum(ext_sub_part_n_s)
  # generate keys
  [key, *subkey_s] = jrandom.split(key, sub_pop_n * ext_sub_pop_n + 1)
  # first for each subpop and each ext subpop generate all instances
  stacked_block_s = [
    [unscaled_ext_mean[sub_pop_idx, ext_sub_pop_idx] / ext_part_n
     + jnp.multiply(
       unscaled_ext_std[sub_pop_idx, ext_sub_pop_idx] / jnp.sqrt(ext_part_n),
       jrandom.normal(subkey_s[sub_pop_idx * sub_pop_n + ext_sub_pop_idx],
                      (ext_connectivity_n,
                       sub_part_n_s[sub_pop_idx], ext_sub_part_n_s[ext_sub_pop_idx])))
     for ext_sub_pop_idx in range(ext_sub_pop_n)]
    for sub_pop_idx in range(sub_pop_n)]
  # then combine blocks
  return([jnp.block(stacked_block_s),
          key])

# phases
def phase_s_generator(ext_part_n, phase_n,
                      key = jnp.array([0, 0], dtype = jnp.uint32)):
  [key, subkey] = jrandom.split(key)
  return([jrandom.uniform(subkey, (phase_n, ext_part_n)),
          key])

# initial conditions
def initial_condition_s_generator(part_n, mean, cov, initial_condition_n,
                                  key = jnp.array([0, 0], dtype = jnp.uint32)):
  [key, subkey] = jrandom.split(key)
  return([jrandom.multivariate_normal(subkey, mean, cov, (initial_condition_n, )),
          key])



# for describing the system
# # nonlinearity
# def nonlinearity(preactivation, bias):
#   return((preactivation <= 0).astype(jnp.float32) * bias * jnp.tanh(preactivation / bias)
#          + (preactivation > 0).astype(jnp.float32) * (2 - bias) * jnp.tanh(preactivation / (2 - bias)))
# nonlinearity = jax.jit(nonlinearity)

# velocity as a function for almost linear networks
def almlin_velocity_fct(connectivity, position):
  return(-1. * position + connectivity @ jnp.tanh(position))
almlin_velocity_fct = jax.jit(almlin_velocity_fct)



# for external inputs
# finding the external conditions at a scalar time
def condition_indicator_s_fct(labeled_time_interval_s, time):
  return(labeled_time_interval_s[1]
         * ((labeled_time_interval_s[0][:, 0] <= time)
            & (time < labeled_time_interval_s[0][:, 1])).astype(jnp.float32))
condition_indicator_s_fct = jax.jit(condition_indicator_s_fct)

# external input as a function for randomly shifted sinusoidal inputs
def sin_ext_input_fct(wave, ext_connectivity, phase, labeled_time_interval_s, time):
  return(
    jnp.multiply(
      jnp.sum(condition_indicator_s_fct(labeled_time_interval_s, time)),
      ext_connectivity
      @ (wave[0] * jnp.cos(2 * jnp.pi * (wave[1] * time + phase)))))
sin_ext_input_fct = jax.jit(sin_ext_input_fct)



# for solving ODEs
# solver for autonomous/isolated ODEs
def rk4_iso_ode_solver(velocity_fct, initial_condition, time_interval, resolution):
  # find the system size, step size, and step number
  part_n = initial_condition.shape[0]
  step_size = 1 / resolution
  step_n = jnp.round((time_interval[1] - time_interval[0]) * resolution).astype(int)
  # initalize output array
  position_s = jnp.zeros((part_n, step_n))
  # start evolving
  temp_position = initial_condition
  for step_idx in range(step_n):
    correction_0 = velocity_fct(temp_position)
    correction_1 = velocity_fct(temp_position + step_size * correction_0 / 2)
    correction_2 = velocity_fct(temp_position + step_size * correction_1 / 2)
    correction_3 = velocity_fct(temp_position + step_size * correction_2)
    temp_position = temp_position + step_size * (correction_0 / 6 + correction_1 / 3
                                                 + correction_2 / 3 + correction_3 / 6)
    position_s = position_s.at[:, step_idx].set(temp_position)
  return(position_s)
rk4_iso_ode_solver = jax.jit(rk4_iso_ode_solver)

# nonautonomous/open ODE solver by reducing to autonomous
def rk4_open_ode_solver(velocity_fct, ext_input_fct, initial_condition,
                        time_interval, resolution):
  # combine self velocity and external input and add time
  def four_net_velocity_fct(four_position):
    return(jnp.concatenate([jnp.atleast_1d(1.),
                            velocity_fct(four_position[1:]) + ext_input_fct(four_position[0])]))
  four_net_velocity_fct = jax.jit(four_net_velocity_fct) # necessary, otherwise 60x longer
  # add time to initial condition
  four_initial_condition = jnp.concatenate([jnp.atleast_1d(time_interval[0]),
                                            initial_condition])
  return(rk4_iso_ode_solver(four_net_velocity_fct, four_initial_condition,
                            time_interval, resolution)[1:])

# solver dealing with multiple intervals
def rk4_open_ode_segmenting_solver(velocity_fct, ext_input_fct, initial_condition,
                                   time_interval_s, resolution):
  # initialize list for each entry (hopefully only) referring to an array
  time_interval_n = time_interval_s.shape[0]
  position_s = [0 for time_interval_idx in range(time_interval_n)]
  # filling the list one by one and updating the initial condition
  temp_initial_condition = initial_condition
  for time_interval_idx in range(time_interval_n):
    position_s[time_interval_idx] = rk4_open_ode_solver(velocity_fct, ext_input_fct,
                                                        temp_initial_condition,
                                                        time_interval_s[time_interval_idx], resolution)
    temp_initial_condition = position_s[time_interval_idx][:, -1]
  return(position_s)



# first two cumulants
# mean
def mean_s_fct(traj_s):
  return(jnp.mean(traj_s, axis = -1))

# mean and cov
def cov_s_fct(traj_s, mean_s):
  # find shapes
  traj_shape = traj_s.shape
  axis_n = len(traj_shape)
  axis_order = tuple(jnp.arange(axis_n - 2)) + (axis_n - 1, axis_n - 2)
  # compute
  mean_s = jnp.expand_dims(mean_s, -1)
  deviation_s = traj_s - mean_s
  cov_s = deviation_s @ jnp.transpose(deviation_s, axis_order) / traj_shape[-1]
  return(cov_s)
cov_s_fct = jax.jit(cov_s_fct)

# cross/lagged-covariance with auto broadcasting (so can do cross
def matched_correlation_fct(traj_1, traj_2):
  # find shapes
  correlation_len = traj_1.shape[-1]
  convolution_len = 2 * correlation_len - 1
  # remove means
  deviation_s_1 = traj_1 - jnp.mean(traj_1, axis = -1,
                                     keepdims = True)
  deviation_s_2 = traj_2 - jnp.mean(traj_2, axis = -1,
                                     keepdims = True)
  # calculate the (unflipped) convolution
  ft_1 = jnp.fft.rfft(deviation_s_1,
                      convolution_len)
  ft_2 = jnp.fft.rfft(deviation_s_2[..., ::-1],
                      convolution_len)
  convolved_array = jnp.fft.irfft(ft_1 * ft_2,
                                  convolution_len)[...,
                                    (correlation_len - 1):]
  # normalize by the number of terms
  return(convolved_array / (jnp.arange(correlation_len)[::-1] + 1))
matched_correlation_fct = jax.jit(matched_correlation_fct)



# participation ratio
# 1d arrays only
def pr_fct(array):
  return(jnp.mean(array) ** 2 / jnp.mean(array ** 2))
# pr_fct = jax.jit(pr_fct)



# getting stats
# prints the format
def low_res_traj_s_fct(connectivity_s, wave_s, ext_connectivity_s, phase_s, initial_condition_s,
                       labeled_time_interval_s, resolution, acor_inclusion = False):
  # find condition numbers and system size
  connectivity_n = connectivity_s.shape[0]
  wave_n = wave_s.shape[0]
  ext_connectivity_n = ext_connectivity_s.shape[0]
  phase_n = phase_s.shape[0]
  initial_condition_n = initial_condition_s.shape[0]
  time_len = ((jnp.sum(jnp.round((labeled_time_interval_s[:, 1]
                                  - labeled_time_interval_s[:, 0])
                                 * resolution))
               - 1) // resolution + 1).astype(int)
  part_n = initial_condition_s.shape[1]
  print([connectivity_n, "connectivity_n"],
        [wave_n, "wave_n"],
        [ext_connectivity_n, "ext_connectivity_n"],
        [phase_n, "phase_n"],
        [initial_condition_n, "initial_condition_n"],
        sep = "\r\n")
  # initialize and loop through conditions
  low_res_traj_s = jnp.zeros(connectivity_n, wave_n, ext_connectivity_n, phase_n,
                     initial_condition_n,
                     part_n, time_len)
  print(round(time.time()))
  for connectivity_idx in range(connectivity_n):
    for wave_idx in range(wave_n):
      for ext_connectivity_idx in range(ext_connectivity_n):
        for phase_idx in range(phase_n):
          for initial_condition_idx in range(initial_condition_n):
            low_res_traj_s = low_res_traj_s.at[
              connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
              initial_condition_n,
              :].set(jnp.tanh(jnp.concatenate(
                rk4_open_ode_segmenting_solver(
                  lambda position: almlin_velocity_fct(
                    connectivity_s[connectivity_idx], position),
                  lambda time: sin_ext_input_fct(
                    wave_s[wave_idx],
                    ext_connectivity_s[ext_connectivity_idx],
                    phase_s[phase_idx],
                    labeled_long_interval_s, time), 
                  initial_condition_s[initial_condition_idx],
                  labeled_time_interval_s[0],
                  resolution), axis = -1)[:, ::resolution]))
    print(round(time.time()), connectivity_idx, sep = ",")
  return(low_res_traj_s)



# covariance orientation similarities
# spectral matrix power for general square matrix
def matrix_power_fct(matrix, power):
  [eva_s, eve_s] = jnp.linalg.eig(matrix)
  return(eve_s @ jnp.diag(eva_s) ** power @ jnp.linalg.inv(eve_s))
matrix_power_fct = jax.jit(matrix_power_fct)

# faster spec mat power for hermitian, returns complex 64
def h_matrix_power_fct(h_matrix, power):
  [eva_s, eve_s] = jnp.linalg.eigh(h_matrix)
  return(eve_s @ jnp.diag(eva_s).astype(complex) ** power
         @ jnp.linalg.inv(eve_s))
h_matrix_power_fct = jax.jit(h_matrix_power_fct)

# faster spec mat power for positive definite, returns float32
def pd_matrix_power_fct(pd_matrix, power):
  [eva_s, eve_s] = jnp.linalg.eigh(pd_matrix)
  # to avoid numerical errors resulting in neg evas
  eva_s = eva_s - jnp.min(eva_s, initial = 0)
  return(eve_s @ jnp.diag(eva_s) ** power @ jnp.linalg.inv(eve_s))
pd_matrix_power_fct = jax.jit(pd_matrix_power_fct)

# orientation similarity, tr(sqrt1 sqrt2)/std1/std2
def ori_similarity_fct(cov_1, cov_2):
  sqrt_cov_1 = pd_matrix_power_fct(cov_1, 0.5)
  sqrt_cov_2 = pd_matrix_power_fct(cov_2, 0.5)
  return(jnp.trace(sqrt_cov_1 @ sqrt_cov_2)
         / jnp.sqrt(jnp.trace(cov_1)) / jnp.sqrt(jnp.trace(cov_2)))
ori_similarity_fct = jax.jit(ori_similarity_fct)

# matched
def matched_cov_similarity_s_fct(cov_s_1, cov_s_2):
  cov_shape = cov_s_1.shape
  matched_shape = cov_shape[:-2]
  matched_size = jnp.prod(jnp.asarray(matched_shape))
  part_n = cov_shape[-1]
  cov_s_1_reshaped = cov_s_1.reshape((matched_size, ) + (part_n, part_n))
  cov_s_2_reshaped = cov_s_2.reshape((matched_size, ) + (part_n, part_n))
  return(jnp.asarray([ori_similarity_fct(cov_s_1_reshaped[cov_idx],
                                         cov_s_2_reshaped[cov_idx])
                      for cov_idx in range(matched_size)]).reshape(matched_shape))
#matched_cov_similarity_s_fct = jax.jit(matched_cov_similarity_s_fct)


# covariance sizes
# using traces (see writeup for justification)
def cov_size_s_fct(cov_s):
  return(jnp.trace(cov_s, axis1=-1, axis2=-2))



# pr_eve_s, pr_nat_s
# # eve_s in numpy are already in columns
# rotation_eve = jnp.linalg.inv(jnp.linalg.eig(connectivity_s[connectivity_idx])[1])



# for analysis and plotting
# load saved npz's as lists
def load_as_list(zipname, allow_pickle = False):
  return([jnp.load(zipname, allow_pickle = allow_pickle)[array_name] 
          for array_name in jnp.load(zipname, allow_pickle = allow_pickle).files])
