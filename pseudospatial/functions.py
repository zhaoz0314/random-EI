import jax # for jax.jit
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
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
def sub_part_n_s_fct(part_n_s, sub_part_r_s):
  # calculate rough numbers
  if jnp.asarray(part_n_s).size > 1:
    reshaped_part_n_s = jnp.swapaxes(jnp.atleast_2d(part_n_s), 0, 1)
    sub_part_n_s = jnp.floor(reshaped_part_n_s * sub_part_r_s)
  else:
    sub_part_n_s = jnp.floor(part_n_s * sub_part_r_s)
  # in case the rounded numbers don't add up
  sub_part_n_s = sub_part_n_s.at[..., -1].set(part_n_s - jnp.sum(sub_part_n_s[..., :-1],
                                                               axis = -1))
  return(sub_part_n_s.astype(int))

# (scaled or unscaled) mean calculator
def ei_mean_balancer(ei_part_n_s, e_mean):
  return(jnp.tile(jnp.asarray([e_mean, - e_mean * ei_part_n_s[0] / ei_part_n_s[1]]), (2, 1)))

# intensive K
def dense_bernoulli_parameter_s_fct(unscaled_mean, unscaled_std):
  in_r = 1 / (1 + unscaled_std ** 2 / unscaled_mean ** 2)
  unscaled_str = unscaled_mean / jnp.sqrt(in_r)
  return([unscaled_str, in_r]) # unscaled mean, std = unscaled_str, 0

# # extensive K
# def sparse_bernoulli_parameter_s_fct(unscaled_mean, unscaled_std, part_n):
#   in_r = 1 / (1 + unscaled_std ** 2 / unscaled_mean ** 2)
#   unscaled_str = unscaled_mean / jnp.sqrt(in_r)
#   return([unscaled_str, in_r]) # unscaled mean, std = unscaled_str, 0

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
    shift_s = jnp.floor(jnp.arange(part_n) * sub_part_r_s[sub_pop_idx]) - closeness_s[sub_pop_idx]
    adjacency_segment_s[sub_pop_idx] = jax.lax.fori_loop(
      0, part_n,
      lambda row_idx, rolled_matrix_s:
      rolled_matrix_s.at[..., row_idx, :].set(jnp.roll(rolled_matrix_s[..., row_idx, :],
                                                       shift_s[row_idx],
                                                       axis = -1)),
      adjacency_segment_s[sub_pop_idx])
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
  step_n = jnp.round((time_interval[1] - time_interval[0]) * resolution
                     + 1).astype(int)
  # evolving positions
  position_s = jnp.zeros((part_n, step_n)).at[:, 0].set(initial_condition)
  def step_forward(step_idx, position_s):
    previous_position = position_s[:, step_idx - 1]
    correction_0 = velocity_fct(previous_position)
    correction_1 = velocity_fct(previous_position + step_size * correction_0 / 2)
    correction_2 = velocity_fct(previous_position + step_size * correction_1 / 2)
    correction_3 = velocity_fct(previous_position + step_size * correction_2)
    next_position = previous_position + step_size * (correction_0 / 6 + correction_1 / 3
                                                 + correction_2 / 3 + correction_3 / 6)
    return(position_s.at[:, step_idx].set(next_position))
  position_s = jax.lax.fori_loop(1, step_n, step_forward, position_s)[:, 1:]
  return(position_s)
# cannot be jitted since output depends on shape, but the evolution is compiled in fori

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
# output depends on shape

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



# getting trajectories
# prints the format
def low_res_traj_s_fct(connectivity_s, wave_s, ext_connectivity_s, phase_s, initial_condition_s,
                       labeled_time_interval_s, resolution, frame_gap):
  # find condition numbers and system size
  connectivity_n = connectivity_s.shape[0]
  wave_n = wave_s.shape[0]
  ext_connectivity_n = ext_connectivity_s.shape[0]
  phase_n = phase_s.shape[0]
  initial_condition_n = initial_condition_s.shape[0]
  frame_n = ((jnp.sum(jnp.round((labeled_time_interval_s[0][1:, 1]
                                 - labeled_time_interval_s[0][1:, 0])
                                * resolution))
              - 1) // frame_gap + 1).astype(int)
  part_n = initial_condition_s.shape[1]
  print([connectivity_n, "connectivity_n"],
        [wave_n, "wave_n"],
        [ext_connectivity_n, "ext_connectivity_n"],
        [phase_n, "phase_n"],
        [initial_condition_n, "initial_condition_n"],
        sep = "\r\n")
  # initialize and loop through conditions
  low_res_traj_s = np.zeros((connectivity_n, wave_n, ext_connectivity_n, phase_n,
                             initial_condition_n,
                             part_n, frame_n))
  for connectivity_idx in range(connectivity_n):
    connectivity_step_start = round(time.time())
    for wave_idx in range(wave_n):
      for ext_connectivity_idx in range(ext_connectivity_n):
        for phase_idx in range(phase_n):
          for initial_condition_idx in range(initial_condition_n):
            low_res_traj_s[connectivity_idx, wave_idx, ext_connectivity_idx, phase_idx,
                           initial_condition_idx
                           ] = jnp.tanh(jnp.concatenate(
                             rk4_open_ode_segmenting_solver(
                               lambda position: almlin_velocity_fct(
                                 connectivity_s[connectivity_idx], position),
                               lambda time: sin_ext_input_fct(
                                 wave_s[wave_idx],
                                 ext_connectivity_s[ext_connectivity_idx],
                                 phase_s[phase_idx],
                                 labeled_time_interval_s, time), 
                               initial_condition_s[initial_condition_idx],
                               labeled_time_interval_s[0],
                               resolution)[1:], axis = -1)[:, ::frame_gap])
    connectivity_step_end = round(time.time())
    print(connectivity_idx,
          "{:.2f} mins".format((connectivity_step_end - connectivity_step_start)
                               / 60),
          sep = ",")
  return(low_res_traj_s)



# first two cumulants
# mean
def mean_s_fct(traj_s):
  return(jnp.mean(traj_s, axis = -1))

# mean and cov
def cov_s_fct(traj_s, mean_s):
  # find shapes
  frame_n = traj_s.shape[-1]
  # compute
  mean_s = jnp.expand_dims(mean_s, -1)
  deviation_s = traj_s - mean_s
  cov_s = deviation_s @ jnp.swapaxes(deviation_s, -1, -2) / frame_n
  return(cov_s)
cov_s_fct = jax.jit(cov_s_fct)

# cross/lagged-covariance with auto broadcasting (so can do cross
def matched_correlation_fct(traj_s_1, traj_s_2, mean_s_1, mean_s_2):
  # find shapes
  frame_n = traj_s_1.shape[-1]
  convolution_len = 2 * frame_n - 1
  # remove means
  mean_s_1 = jnp.expand_dims(mean_s_1, -1)
  mean_s_2 = jnp.expand_dims(mean_s_2, -1)
  deviation_s_1 = traj_s_1 - mean_s_1
  deviation_s_2 = traj_s_2 - mean_s_2
  # calculate the (unflipped) convolution
  ft_1 = jnp.fft.rfft(deviation_s_1,
                      convolution_len)
  ft_2 = jnp.fft.rfft(deviation_s_2[..., ::-1],
                      convolution_len)
  convolved_array = jnp.fft.irfft(ft_1 * ft_2,
                                  convolution_len)[...,
                                    (frame_n - 1):]
  # normalize by the number of terms
  return(convolved_array / (jnp.arange(frame_n)[::-1] + 1))
matched_correlation_fct = jax.jit(matched_correlation_fct)



# cov-based stats
# eigen system/principal components
def es_s_fct(matrix_s, prop = "pd"):
  if prop == "pd":
    output = jnp.linalg.eigh(matrix_s)
    # remove numerical errors
    output = [output[0] - jnp.min(output[0], initial = 0,
                                  axis = -1,
                                  keepdims = True),
              output[1]]
  elif prop == "h":
    output = jnp.linalg.eigh(matrix_s)
  else:
    output = jnp.linalg.eig(matrix_s)
  return(output)
es_s_fct = jax.jit(es_s_fct, static_argnums = (1,))

# take leading pcs when within tolerance
def es_s_thinner(es_s, dim_r, tolerance):
  dim_n_full = es_s[0].shape[-1]
  dim_n_thin = jnp.round(es_s[0].shape[-1] * dim_r).astype(int)
  var_s_full = jnp.sum(es_s[0], axis = -1)
  var_s_thin = jnp.sum(es_s[0][..., (dim_n_full - dim_n_thin):], axis = -1)
  error_r_s = (var_s_full - var_s_thin) / var_s_full
  if jnp.prod(error_r_s < tolerance):
    thin_es_s = [es_s[0][..., (dim_n_full - dim_n_thin):],
                 es_s[1][..., (dim_n_full - dim_n_thin):]]
  else:
    print(
      "error ({0:.4f}) exceeds tolerance ({1:.4f})".format(
        jnp.max(error_r_s), jnp.min(tolerance)))
    thin_es_s = es_s
  return(thin_es_s)

# participation ratios
def dim_r_s_fct(var_s):
  return(jnp.mean(var_s, axis = -1) ** 2 / jnp.mean(var_s ** 2, axis = -1))
dim_r_s_fct = jax.jit(dim_r_s_fct)

# traces/attractor sizes
def size_s_fct(var_s):
  return(jnp.mean(var_s, axis = -1))

# orientation similarity, tr(sqrt1 sqrt2)/std1/std2
def ori_similarity_s_fct(es_s_1, es_s_2, size_s_1, size_s_2):
  std_s_1 = jnp.sqrt(es_s_1[0])
  std_s_2 = jnp.sqrt(es_s_2[0])
  rot_s = jnp.swapaxes(es_s_1[1], -1, -2) @ es_s_2[1]
  return(jnp.einsum("...i, ...ij, ...ij, ...j",
                    std_s_1, rot_s, rot_s, std_s_2,
                    optimize = True)
         / jnp.sqrt(size_s_1 * size_s_2))
ori_similarity_s_fct = jax.jit(ori_similarity_s_fct)



# pr_eve_s, pr_nat_s
# # eve_s in numpy are already in columns
# rotation_eve = jnp.linalg.inv(jnp.linalg.eig(connectivity_s[connectivity_idx])[1])



# for analysis and plotting
# load saved npz's as lists
def load_as_list(zipname, allow_pickle = False):
  return([jnp.load(zipname, allow_pickle = allow_pickle)[array_name] 
          for array_name in jnp.load(zipname, allow_pickle = allow_pickle).files])
