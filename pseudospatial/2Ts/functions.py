import numpy as np
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
  sub_part_n_s = np.floor(part_n * sub_part_r_s)
  # in case the rounded numbers don't add up
  sub_part_n_s[-1] = part_n - jnp.sum(sub_part_n_s[:-1])
  return(sub_part_n_s.astype(int))

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
    np.concatenate([jrandom.bernoulli(subkey_s[sub_pop_idx * 2 + 0], nearby_prob,
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
      adjacency_segment_s[sub_pop_idx][
        :, row_idx] = jnp.roll(adjacency_segment_s[sub_pop_idx][:, row_idx],
                               jnp.floor(row_idx * sub_part_r_s[sub_pop_idx])
                               - closeness_s[sub_pop_idx],
                               axis = -1)
  adjacency_s = jnp.concatenate(adjacency_segment_s, axis = -1, dtype = bool)
  # sparsify and rescale by in_part_r * part_n
  connectivity_s = connectivity_s * adjacency_s.astype(int) / jnp.sqrt(in_part_r)
  # only remove row mean (ignoring zeros) for nonzero entries
  connectivity_s = (connectivity_s
                    - adjacency_s.astype(int) * jnp.mean(
                      connectivity_s, axis = -1, where = adjacency_s,
                      keepdims = True))
  return([connectivity_s, key])

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

# time intervals
# labeled short intervals
def labeled_short_interval_s_fct(tot_interval, condition_s, short_interval_n_in_long_s):
  short_interval_n = jnp.sum(short_interval_n_in_long_s)
  short_interval_len = ((tot_interval[1] - tot_interval[0]) / short_interval_n)
  labeled_short_interval_s = [jnp.transpose(jnp.tile(jnp.linspace(tot_interval[0],
                                                                  tot_interval[1],
                                                                  short_interval_n + 1)[:-1],
                                                     (2, 1)))
                              + jnp.asarray([0, short_interval_len]),
                              jnp.concatenate(
                                [jnp.full((short_interval_n_in_long_s[long_interval_idx], ),
                                          condition_s[long_interval_idx])
                                 for long_interval_idx
                                 in range(short_interval_n_in_long_s.shape[0])])]
  return(labeled_short_interval_s)

# labeled long intervals
def labeled_long_interval_s_fct(tot_interval, condition_s, short_interval_n_in_long_s):
  short_interval_len = ((tot_interval[1] - tot_interval[0]) / jnp.sum(short_interval_n_in_long_s))
  time_tick_s = tot_interval[0] + (jnp.asarray([jnp.sum(short_interval_n_in_long_s[:long_interval_idx])
                                                for long_interval_idx
                                                in range(short_interval_n_in_long_s.shape[0] + 1)])
                                   * short_interval_len)
  labeled_long_interval_s = [jnp.transpose(jnp.asarray([time_tick_s[:-1], time_tick_s[1:]])),
                              condition_s]
  return(labeled_long_interval_s)

# long intervals in terms of short interval indices
def rel_long_interval_s_fct(long_interval_s, short_interval_len):
  return(((long_interval_s - long_interval_s[0, 0]) / short_interval_len).astype(int))



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
  position_s = np.zeros((part_n, step_n), dtype = np.float32)
  # start evolving
  temp_position = initial_condition
  for step_idx in range(step_n):
    correction_0 = velocity_fct(temp_position)
    correction_1 = velocity_fct(temp_position + step_size * correction_0 / 2)
    correction_2 = velocity_fct(temp_position + step_size * correction_1 / 2)
    correction_3 = velocity_fct(temp_position + step_size * correction_2)
    temp_position = temp_position + step_size * (correction_0 / 6 + correction_1 / 3
                                                 + correction_2 / 3 + correction_3 / 6)
    position_s[:, step_idx] = temp_position
  return(position_s)

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



# for obtaining correlations
# cross/lagged-covariance with auto broadcasting (so can do cross
def matched_correlation_fct(array_1, array_2):
  # find shapes
  correlation_len = array_1.shape[-1]
  convolution_len = 2 * correlation_len - 1
  # remove means
  deviation_s_1 = array_1 - jnp.mean(array_1, axis = -1,
                                     keepdims = True)
  deviation_s_2 = array_2 - jnp.mean(array_2, axis = -1,
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


# correlation with or wo mean removal, displacing last axis
def correlation_fct(array_1, array_2, lag_s, mean_removal = True, cor_not_cov = False):
  # initialize output array
  correlation_s = np.zeros(array_1.shape[:-1] + (lag_s.shape[0], ), dtype = np.float32)
  # filling the array
  if mean_removal == True:
    # find repeatedly used variables
    mean_1 = jnp.mean(array_1, axis = -1, keepdims = True)
    mean_2 = jnp.mean(array_2, axis = -1, keepdims = True)
    if cor_not_cov == False:
      std_1 = 1
      std_2 = 1
    elif cor_not_cov == True:
      std_1 = jnp.std(array_1, axis = -1)
      std_2 = jnp.std(array_2, axis = -1)
    for lag_idx in range(lag_s.shape[0]):
      lag = lag_s[lag_idx]
      if lag == 0:
        array_1_shorten_shifted = array_1 - mean_1
        array_2_shorten_shifted = array_2 - mean_2
      elif lag > 0:
        array_1_shorten_shifted = array_1[..., lag:] - mean_1
        array_2_shorten_shifted = array_2[..., :-lag] - mean_2
      elif lag < 0:
        array_1_shorten_shifted = array_1[..., :lag] - mean_1
        array_2_shorten_shifted = array_2[..., -lag:] - mean_2
      correlation_s[..., lag_idx] = jnp.mean(array_1_shorten_shifted
                                                      * jnp.conj(array_2_shorten_shifted),
                                                      axis = -1) / std_1 / std_2
  elif mean_removal == False:
    # find repeatedly used variables
    if cor_not_cov == False:
      std_1 = 1
      std_2 = 1
    elif cor_not_cov == True:
      unit_norm_1 = jnp.sqrt(jnp.mean(array_1 * jnp.conj(array_1), axis = -1))
      unit_norm_2 = jnp.sqrt(jnp.mean(array_2 * jnp.conj(array_2), axis = -1))
    for lag_idx in range(lag_s.shape[0]):
      lag = lag_s[lag_idx]
      if lag == 0:
        array_1_shorten = array_1
        array_2_shorten = array_2
      elif lag > 0:
        array_1_shorten = array_1[..., lag:]
        array_2_shorten = array_2[..., :-lag]
      elif lag < 0:
        array_1_shorten = array_1[..., :lag]
        array_2_shorten = array_2[..., -lag:]
      correlation_s[..., lag_idx] = (jnp.mean(array_1_shorten * jnp.conj(array_2_shorten),
                                                      axis = -1)
                                              / unit_norm_1 / unit_norm_2)
  return(correlation_s)



# participation ratio
# 1d arrays only
def pr_fct(array):
  return(jnp.mean(array) ** 2 / jnp.mean(array ** 2))
# pr_fct = jax.jit(pr_fct)



# getting stats
# prints the format
def network_stat_s_fct(connectivity_s, wave_s, ext_connectivity_s, phase_s, initial_condition_s,
                       time_interval_info, resolution, acor_inclusion = False):
  # find condition numbers (with interval conditions)
  connectivity_n = connectivity_s.shape[0]
  wave_n = wave_s.shape[0]
  ext_connectivity_n = ext_connectivity_s.shape[0]
  phase_n = phase_s.shape[0]
  initial_condition_n = initial_condition_s.shape[0]
  part_n = initial_condition_s.shape[1]
  [tot_interval, condition_s, short_interval_n_in_long_s] = time_interval_info
  labeled_short_interval_s = labeled_short_interval_s_fct(tot_interval,
                                                          condition_s,
                                                          short_interval_n_in_long_s)
  labeled_long_interval_s = labeled_long_interval_s_fct(tot_interval,
                                                        condition_s,
                                                        short_interval_n_in_long_s)
  short_interval_n = labeled_short_interval_s[0].shape[0]
  long_interval_n = labeled_long_interval_s[0].shape[0]
  # find the system size, int interval lengths, and (relative) long interval ticks
  short_interval_len = labeled_short_interval_s[0][0, 1] - labeled_short_interval_s[0][0, 0]
  long_interval_len = labeled_long_interval_s[0][:, 1] - labeled_long_interval_s[0][:, 0]
  int_short_interval_len = jnp.floor(short_interval_len).astype(int)
  min_int_long_interval_len = jnp.floor(jnp.min(long_interval_len)).astype(int)
  short_step_n = jnp.round(short_interval_len * resolution).astype(int)
  rel_long_interval_s = rel_long_interval_s_fct(labeled_long_interval_s[0],
                                                short_interval_len)
  # initialize output arrays
  mean_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                     initial_condition_n, short_interval_n,
                     part_n),
                    dtype = np.float32)
  cov_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                    initial_condition_n, short_interval_n,
                    part_n, part_n),
                   dtype = np.float32)
  short_pr_pca_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                       initial_condition_n, short_interval_n),
                      dtype = np.float32)
  long_pr_pca_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                       initial_condition_n, long_interval_n),
                      dtype = np.float32)
  mean_traj_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                          initial_condition_n, short_interval_n,
                          short_step_n),
                         dtype = np.float32)
  if acor_inclusion == True:
    short_acor_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                             initial_condition_n, short_interval_n,
                             int_short_interval_len),
                            dtype = np.float32)
    long_acor_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                            initial_condition_n, long_interval_n,
                            min_int_long_interval_len),
                           dtype = np.float32)
  print([connectivity_n, "connectivity_n"],
        [wave_n, "wave_n"],
        [ext_connectivity_n, "ext_connectivity_n"],
        [phase_n, "phase_n"],
        [initial_condition_n, "initial_condition_n"],
        [short_interval_n, "short_interval_n"],
        sep = "\r\n")
  if acor_inclusion == False:
    print(["mean_s", "cov_s", "short_pr_pca_s", "long_pr_pca_s", "mean_traj_s"])
  elif acor_inclusion == True:
    print(["mean_s", "cov_s", "short_pr_pca_s", "long_pr_pca_s", "mean_traj_s", "short_acor_s", "long_acor_s"])
  # loop through conditions
  print(round(time.time()))
  for connectivity_idx in range(connectivity_n):
    for wave_idx in range(wave_n):
      for ext_connectivity_idx in range(ext_connectivity_n):
        for phase_idx in range(phase_n):
          for initial_condition_idx in range(initial_condition_n):
            temp_traj = [jnp.tanh(short_interval_preactivation)
                         for short_interval_preactivation in rk4_open_ode_segmenting_solver(
                             lambda position: almlin_velocity_fct(
                               connectivity_s[connectivity_idx], position),
                             lambda time: sin_ext_input_fct(
                               wave_s[wave_idx],
                               ext_connectivity_s[ext_connectivity_idx],
                               phase_s[phase_idx],
                               labeled_long_interval_s, time), 
                             initial_condition_s[initial_condition_idx],
                             labeled_short_interval_s[0],
                             resolution)]
            # short interval-specific stats
            for short_interval_idx in range(short_interval_n):
              # stats independent of cov
              mean_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                     initial_condition_idx, short_interval_idx,
                     :] = jnp.mean(temp_traj[short_interval_idx], axis = 1)
              mean_traj_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                          initial_condition_idx, short_interval_idx,
                          :] = jnp.mean(temp_traj[short_interval_idx], axis = 0)
              # stats dependent on cov
              temp_cov = (temp_traj[short_interval_idx] @ jnp.transpose(temp_traj[short_interval_idx])
                          / short_step_n)
              cov_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                    initial_condition_idx, short_interval_idx,
                    :, :] = temp_cov
              short_pr_pca_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                             initial_condition_idx, short_interval_idx
                             ] = pr_fct(jnp.linalg.eigh(temp_cov)[0])
              if acor_inclusion == True:
                short_acor_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                             initial_condition_idx, short_interval_idx,
                             :] = jnp.mean(correlation_fct(
                               temp_traj[short_interval_idx], temp_traj[short_interval_idx],
                               jnp.arange(int_short_interval_len) * resolution), axis = 0)
            # long interval specific stats
            for long_interval_idx in range(long_interval_n):
              long_pr_pca_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                            initial_condition_idx, long_interval_idx
                            ] = pr_fct(jnp.linalg.eigh(jnp.mean(
                              cov_s[connectivity_idx, wave_idx,
                                    ext_connectivity_idx, phase_idx,
                                    initial_condition_idx,
                                    :, :, :],
                              axis = 0))[0])
              if acor_inclusion == True:
                temp_concatenated_traj = jnp.concatenate(
                  temp_traj[rel_long_interval_s[long_interval_idx, 0]
                            :rel_long_interval_s[long_interval_idx, 1]], axis = -1)
                long_acor_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                            initial_condition_idx, long_interval_idx,
                            :] = jnp.mean(correlation_fct(
                              temp_concatenated_traj, temp_concatenated_traj,
                              jnp.arange(min_int_long_interval_len) * resolution), axis = 0)
    print(round(time.time()), connectivity_idx, sep = ",")
  if acor_inclusion == False:
    output_list = [mean_s, cov_s, short_pr_pca_s, long_pr_pca_s, mean_traj_s]
  elif acor_inclusion == True:
    output_list = [mean_s, cov_s, short_pr_pca_s, long_pr_pca_s, mean_traj_s,
                   short_acor_s, long_acor_s]
  return(output_list)

# checking affect of resolution
def resolution_effect_fct(connectivity_s, wave_s, ext_connectivity_s, phase_s, initial_condition_s,
                          time_interval_info, resolution):
  # find condition numbers (with interval conditions)
  connectivity_n = connectivity_s.shape[0]
  wave_n = wave_s.shape[0]
  ext_connectivity_n = ext_connectivity_s.shape[0]
  phase_n = phase_s.shape[0]
  initial_condition_n = initial_condition_s.shape[0]
  part_n = initial_condition_s.shape[1]
  [tot_interval, condition_s, short_interval_n_in_long_s] = time_interval_info
  labeled_short_interval_s = labeled_short_interval_s_fct(tot_interval,
                                                          condition_s,
                                                          short_interval_n_in_long_s)
  labeled_long_interval_s = labeled_long_interval_s_fct(tot_interval,
                                                        condition_s,
                                                        short_interval_n_in_long_s)
  short_interval_n = labeled_short_interval_s[0].shape[0]
  long_interval_n = labeled_long_interval_s[0].shape[0]
  # find the system size, int interval lengths, and (relative) long interval ticks
  short_interval_len = labeled_short_interval_s[0][0, 1] - labeled_short_interval_s[0][0, 0]
  long_interval_len = labeled_long_interval_s[0][:, 1] - labeled_long_interval_s[0][:, 0]
  int_short_interval_len = jnp.floor(short_interval_len).astype(int)
  min_int_long_interval_len = jnp.floor(jnp.min(long_interval_len)).astype(int)
  short_step_n = jnp.round(short_interval_len * resolution).astype(int)
  rel_long_interval_s = rel_long_interval_s_fct(labeled_long_interval_s[0],
                                                short_interval_len)
  # initialize output arrays
  mean_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                     initial_condition_n, short_interval_n, 2),
                    dtype = np.float32)
  pr_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                     initial_condition_n, short_interval_n, 2),
                    dtype = np.float32)
  tr_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                     initial_condition_n, short_interval_n, 2),
                    dtype = np.float32)
  ori_sim_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                     initial_condition_n, short_interval_n),
                    dtype = np.float32)
  print([connectivity_n, "connectivity_n"],
        [wave_n, "wave_n"],
        [ext_connectivity_n, "ext_connectivity_n"],
        [phase_n, "phase_n"],
        [initial_condition_n, "initial_condition_n"],
        [short_interval_n, "short_interval_n"],
        sep = "\r\n")
  print(["mean", "pr", "tr", "ori"])
  # loop through conditions
  print(round(time.time()))
  for connectivity_idx in range(connectivity_n):
    for wave_idx in range(wave_n):
      for ext_connectivity_idx in range(ext_connectivity_n):
        for phase_idx in range(phase_n):
          for initial_condition_idx in range(initial_condition_n):
            temp_traj = [jnp.tanh(short_interval_preactivation)
                         for short_interval_preactivation in rk4_open_ode_segmenting_solver(
                             lambda position: almlin_velocity_fct(
                               connectivity_s[connectivity_idx], position),
                             lambda time: sin_ext_input_fct(
                               wave_s[wave_idx],
                               ext_connectivity_s[ext_connectivity_idx],
                               phase_s[phase_idx],
                               labeled_long_interval_s, time), 
                             initial_condition_s[initial_condition_idx],
                             labeled_short_interval_s[0],
                             resolution)]
            low_res_temp_traj = [short_interval_traj[:, ::resolution]
                                 for short_interval_traj in temp_traj]
            # only do short interval-specific stats
            for short_interval_idx in range(short_interval_n):
              # stats independent of cov
              mean_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                         initial_condition_idx, short_interval_idx, 0] = jnp.mean(temp_traj[short_interval_idx])
              mean_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                         initial_condition_idx, short_interval_idx, 1] = jnp.mean(low_res_temp_traj[short_interval_idx])
              # stats dependent on cov
              temp_cov = (temp_traj[short_interval_idx] @ jnp.transpose(temp_traj[short_interval_idx])
                          / short_step_n)
              temp_low_res_cov = (low_res_temp_traj[short_interval_idx] @ jnp.transpose(low_res_temp_traj[short_interval_idx])
                                  / short_interval_len)
              pr_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                   initial_condition_idx, short_interval_idx, 0] = pr_fct(jnp.linalg.eigh(temp_cov)[0])
              pr_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                   initial_condition_idx, short_interval_idx, 1] = pr_fct(jnp.linalg.eigh(temp_low_res_cov)[0])
              tr_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                   initial_condition_idx, short_interval_idx, 0] = jnp.trace(temp_cov)
              tr_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                   initial_condition_idx, short_interval_idx, 1] = jnp.trace(temp_low_res_cov)
              ori_sim_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                        initial_condition_idx, short_interval_idx] = ori_similarity_fct(temp_cov, temp_low_res_cov)
    print(round(time.time()), connectivity_idx, sep = ",")
  return([mean_s, pr_s, tr_s, ori_sim_s])



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

# all pairs within a stack without flattening covs
def self_ori_similarity_s_fct(cov_s):
  # find possible indices
  shape = cov_s.shape[:-2]
  idx_s = [(connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
              initial_condition_idx, time_interval_idx)
             for connectivity_idx in range(shape[0])
             for wave_idx in range(shape[3])
             for ext_connectivity_idx in range(shape[1])
             for phase_idx in range(shape[2])
             for initial_condition_idx in range(shape[4])
             for time_interval_idx in range(shape[5])]
  # get 1d indices
  flattened_idx_s = jnp.arange(jnp.prod(jnp.asarray(shape))).reshape(shape)
  # initialize output
  self_ori_similarity_s = np.zeros(shape + shape, dtype = np.float32)
  # fill in upper half
  for idx_1 in idx_s:
    for idx_2 in idx_s:
      # the entires cannot be copied since it's unclear which one is reached first
      if flattened_idx_s[idx_1] < flattened_idx_s[idx_2]:
        self_ori_similarity_s[idx_1 + idx_2] = ori_similarity_fct(cov_s[idx_1], cov_s[idx_2])
      elif flattened_idx_s[idx_1] == flattened_idx_s[idx_2]:
        self_ori_similarity_s[idx_1 + idx_2] = 1
  return(self_ori_similarity_s)

# reorder, flatten, (and symmetrize) cov similarities
def self_ori_similarity_s_ordered_flattener(self_ori_similarity_s,
                                            order = (0, 5, 1, 2, 3, 4)):
  dim_n = len(order)
  doubled_order = order + tuple((jnp.asarray(order) + dim_n).tolist())
  condition_n = jnp.prod(jnp.asarray(self_ori_similarity_s.shape[:dim_n]))
  self_ori_similarity_s_ordered_flattened = jnp.transpose(
    self_ori_similarity_s, doubled_order).reshape(condition_n, condition_n)
  return(self_ori_similarity_s_ordered_flattened
         + jnp.transpose(self_ori_similarity_s_ordered_flattened)
         - jnp.diag(jnp.diag(self_ori_similarity_s_ordered_flattened)))
self_ori_similarity_s_ordered_flattener = jax.jit(
  self_ori_similarity_s_ordered_flattener)

# all pairs between arrays
def cross_ori_similarity_s_fct(cov_s_1, cov_s_2):
  # find possible indices
  shape_s = [cov_s_1.shape[:6], cov_s_2.shape[:6]]
  idx_s = [[(connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
             initial_condition_idx, time_interval_idx)
            for connectivity_idx in range(shape_s[shape_idx][0])
            for wave_idx in range(shape_s[shape_idx][3])
            for ext_connectivity_idx in range(shape_s[shape_idx][1])
            for phase_idx in range(shape_s[shape_idx][2])
            for initial_condition_idx in range(shape_s[shape_idx][4])
            for time_interval_idx in range(shape_s[shape_idx][5])]
           for shape_idx in range(2)]
  # initialize output
  cross_ori_similarity_s = np.zeros(shape_s[0] + shape_s[1], dtype = np.float32)
  # fill in upper half
  for idx_1 in idx_s[0]:
    for idx_2 in idx_s[1]:
      cross_ori_similarity_s[idx_1 + idx_2] = ori_similarity_fct(cov_s[idx_1], cov_s[idx_2])
  return(cross_ori_similarity_s)

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
