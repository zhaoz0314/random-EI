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



############################## obtaining deterministic conditions ##############################
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
  return(jnp.tile(jnp.array([e_mean, - e_mean * ei_part_n_s[0] / ei_part_n_s[1]]), (2, 1)))

# intensive K # (unscaled mean, std) = (unscaled_strength, 0)
def dense_bernoulli_parameter_s_fct(unscaled_mean, unscaled_std):
  in_r = 1 / (1 + unscaled_std ** 2 / unscaled_mean ** 2)
  unscaled_strength = unscaled_mean / jnp.sqrt(in_r)
  return([unscaled_strength, in_r])

# extensive K # (unscaled mean, std) = (unscaled_strength, 0) # in_r = in_n / part_n
def sparse_bernoulli_parameter_s_fct(unscaled_mean, unscaled_std):
  if unscaled_mean == 0:
    unscaled_strength = unscaled_std
  else:
    print("unscaled_mean not 0")
  return(unscaled_strength)



############################## obtaining random conditions ##############################
# connectivities
def connectivity_s_generator(sub_part_n_s, unscaled_mean, unscaled_std, unscaled_std_log_std,
                             connectivity_n,
                             key = jnp.array([0, 0], dtype = jnp.uint32)):
  # find particle number and arrangement of blocks
  part_n = jnp.sum(sub_part_n_s)
  block_shape = unscaled_mean.shape
  block_n = unscaled_mean.size
  row_gain_n = unscaled_std_log_std.shape[0]
  # split key
  [key, *subkey_s] = jrandom.split(key, block_shape[1] + block_n + 1)
  # first for each block position generate column gains
  unscaled_std_log_s = [
    jnp.expand_dims(unscaled_std_log_std[:, row_idx], 1)
    * jrandom.normal(subkey_s[row_idx],
                     (connectivity_n, sub_part_n_s[row_idx], row_gain_n))
    for row_idx in range(block_shape[0])]
  # then for each block position generate all connectivities
  stacked_block_s = [
    [unscaled_mean[row_idx, column_idx] / jnp.sqrt(part_n)
     + (((unscaled_std[row_idx, column_idx] / jnp.sqrt(part_n)) 
         * jnp.expand_dims(jnp.exp(unscaled_std_log_s[row_idx][..., column_idx]), 2))
        * jrandom.normal(subkey_s[block_shape[1] + row_idx * block_shape[0] + column_idx],
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
                                 unscaled_ext_mean, unscaled_ext_std, inter_instance_correlation,
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
    [unscaled_ext_mean[sub_pop_idx, ext_sub_pop_idx] / ext_part_n # sparse scaling
     + (unscaled_ext_std[sub_pop_idx, ext_sub_pop_idx] / jnp.sqrt(ext_part_n)
        * jnp.transpose(
          jrandom.multivariate_normal(
            subkey_s[sub_pop_idx * sub_pop_n + ext_sub_pop_idx],
            jnp.full((ext_connectivity_n, ), 0),
            inter_instance_correlation,
            (sub_part_n_s[sub_pop_idx], ext_sub_part_n_s[ext_sub_pop_idx]), 
            method = "svd"),
          (2, 0, 1)))
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
def init_condition_s_generator(part_n, mean, cov, init_condition_n,
                               key = jnp.array([0, 0], dtype = jnp.uint32)):
  [key, subkey] = jrandom.split(key)
  return([jrandom.multivariate_normal(subkey, mean, cov, (init_condition_n, )),
          key])



############################## describing the model ##############################
# # nonlinearity of system
# def nonlinearity(preactivation, bias):
#   return(
#     (preactivation <= 0).astype(jnp.float32) * bias * jnp.tanh(preactivation / bias)
#     + (preactivation > 0).astype(jnp.float32) * (2 - bias) * jnp.tanh(preactivation / (2 - bias)))
# nonlinearity = jax.jit(nonlinearity)

# autonomous velocity for almost linear networks
def almlin_velocity_fct(connectivity, position):
  return(-1. * position + connectivity @ jnp.tanh(position))
almlin_velocity_fct = jax.jit(almlin_velocity_fct)

# external input as a function for randomly shifted sinusoidal inputs
def sin_ext_input_fct(wave, ext_connectivity, phase, labeled_time_interval_s, time):
  # indicate presence of nonautonomous velocity at the given time
  condition_indicator = jnp.sum(
    # if intervals contribute
    labeled_time_interval_s[1]
    # if in intervals
    * ((labeled_time_interval_s[0][:, 0] <= time)
       & (time < labeled_time_interval_s[0][:, 1])).astype(jnp.float32))
  return(condition_indicator
         * (ext_connectivity
            @ (wave[0] * jnp.cos(2 * jnp.pi * (wave[1] * time + phase)))))
sin_ext_input_fct = jax.jit(sin_ext_input_fct)



############################## solving ODEs ##############################
# array for containing trajectories
def traj_initializer(init_condition, time_interval, resolution):
  step_n = jnp.round((time_interval[1] - time_interval[0]) * resolution
                     + 1).astype(int)
  traj_holder = jnp.zeros(init_condition.shape + (step_n, )).at[..., 0].set(init_condition)
  return(traj_holder)

# solver for possibly non-autonomous ODEs
def rk4_ode_solver(velocity_fct, ext_input_fct, traj_holder, init_time, resolution):
  # find the system size, step size, and step number
  step_size = 1 / resolution
  step_n = traj_holder.shape[-1]
  # evolving positions
  def step_forward(step_idx, position_s):
    previous_time = init_time + step_size * (step_idx - 1)
    previous_position = position_s[..., step_idx - 1]
    velocity_0 = (velocity_fct(previous_position)
                  + ext_input_fct(previous_time))
    velocity_1 = (velocity_fct(previous_position + step_size * velocity_0 / 2)
                  + ext_input_fct(previous_time + step_size / 2))
    velocity_2 = (velocity_fct(previous_position + step_size * velocity_1 / 2)
                  + ext_input_fct(previous_time + step_size / 2))
    velocity_3 = (velocity_fct(previous_position + step_size * velocity_2)
                  + ext_input_fct(previous_time + step_size))
    next_position = previous_position + step_size * (velocity_0 / 6 + velocity_1 / 3
                                                 + velocity_2 / 3 + velocity_3 / 6)
    return(position_s.at[..., step_idx].set(next_position))
  position_s = jax.lax.fori_loop(1, step_n, step_forward, traj_holder)[..., 1:]
  return(position_s)
# rk4_ode_solver = jax.jit(rk4_ode_solver)



############################## simulate over conditions ##############################
# find arbitrary stats for each condition
def stat_s_fct(connectivity_s, wave_s, ext_connectivity_s, phase_s,
               init_condition_s,
               labeled_time_interval_s, resolution,
               stat_fct, stat_s_holder):
  # find condition numbers
  connectivity_n = connectivity_s.shape[0]
  wave_n = wave_s.shape[0]
  ext_connectivity_n = ext_connectivity_s.shape[0]
  phase_n = phase_s.shape[0]
  init_condition_n = init_condition_s.shape[0]
  condition_n_s = jnp.array([connectivity_n, wave_n, ext_connectivity_n, phase_n,
                             init_condition_n])
  condition_n = jnp.prod(condition_n_s)
  condition_s = [connectivity_s, wave_s, ext_connectivity_s, phase_s,
                 init_condition_s]
  # find stat number
  stat_n = len(stat_s_holder)
  # print
  print([connectivity_n, "connectivity_n"],
        [wave_n, "wave_n"],
        [ext_connectivity_n, "ext_connectivity_n"],
        [phase_n, "phase_n"],
        [init_condition_n, "init_condition_n"],
        sep = "\r\n")
  print("{} stats".format(stat_n))
  # define body function (traj holder created outside to avoid jit)
  joined_time_interval = jnp.array([labeled_time_interval_s[0][0,0],
                                    labeled_time_interval_s[0][-1,1]])
  temp_traj_holder = traj_initializer(init_condition_s[0], joined_time_interval, resolution)
  def stat_s_updater(condition_idx, stat_s_with_condition_s):
    # unpack second variable
    [stat_s, condition_s] = stat_s_with_condition_s
    [connectivity_s, wave_s, ext_connectivity_s, phase_s,
     init_condition_s] = condition_s
    connectivity_n = connectivity_s.shape[0]
    wave_n = wave_s.shape[0]
    ext_connectivity_n = ext_connectivity_s.shape[0]
    phase_n = phase_s.shape[0]
    init_condition_n = init_condition_s.shape[0]
    condition_n_s = jnp.array([connectivity_n, wave_n, ext_connectivity_n, phase_n,
                               init_condition_n])
    # find condition indices
    unraveled_idx = jnp.unravel_index(condition_idx, condition_n_s)
    # simulate traj
    temp_traj = temp_traj_holder.at[..., 0].set(init_condition_s[unraveled_idx[4]])
    temp_traj = jnp.tanh(rk4_ode_solver(
      lambda position:
      almlin_velocity_fct(connectivity_s[unraveled_idx[0]], position),
      lambda time:
      sin_ext_input_fct(
        wave_s[unraveled_idx[1]], ext_connectivity_s[unraveled_idx[2]], phase_s[unraveled_idx[3]],
        labeled_time_interval_s, time),
      temp_traj,
      joined_time_interval[0], resolution))
    # find stats
    temp_stat_s = stat_fct(temp_traj)
    # fill in stats using condition indices
    for stat_idx in range(stat_n):
      stat_s[stat_idx] = stat_s[stat_idx].at[unraveled_idx].set(temp_stat_s[stat_idx])
    return([stat_s, condition_s])
  # run for one connectivity after compilation, time it, and print expected time
  stat_s = jax.lax.fori_loop(0, 1,
                             stat_s_updater, [stat_s_holder, condition_s])[0]
  start_time = round(time.time())
  stat_s = jax.lax.fori_loop(1, 1 + init_condition_n,
                             stat_s_updater, [stat_s, condition_s])[0]
  end_time = round(time.time())
  print("expecting {:.2f} mins".format(
    (end_time - start_time) * condition_n / init_condition_n / 60))
  # run the rest
  stat_s = jax.lax.fori_loop(1 + init_condition_n, condition_n,
                             stat_s_updater, [stat_s, condition_s])[0]
  return(stat_s)



############################## computing stats from traj ##############################
# first two cumulants
# mean
def mean_s_fct(traj_s):
  return(jnp.mean(traj_s, axis = -1))

# cov
def cov_s_fct(traj_s, mean_s):
  # find number of data points
  frame_n = traj_s.shape[-1]
  # compute
  mean_s = jnp.expand_dims(mean_s, -1)
  fluct_s = traj_s - mean_s
  cov_s = fluct_s @ jnp.swapaxes(fluct_s, -1, -2) / frame_n
  return(cov_s)
cov_s_fct = jax.jit(cov_s_fct)

# filtered/windowed cov
def weighted_cov_s_fct(traj_s, mean_s, weight_s):
  # compute
  mean_s = jnp.expand_dims(mean_s, -1)
  fluct_s = traj_s - mean_s
  cov_s = (fluct_s @ jnp.swapaxes(fluct_s * weight_s, -1, -2)
           / jnp.sum(weight_s))
  return(cov_s)
weighted_cov_s_fct = jax.jit(weighted_cov_s_fct)

# cross/lagged-covariance with auto broadcasting (spatial cross possible
def matched_correlation_fct(traj_s_1, traj_s_2, mean_s_1, mean_s_2):
  # find shapes
  frame_n = traj_s_1.shape[-1]
  convolution_len = 2 * frame_n - 1
  # remove means
  mean_s_1 = jnp.expand_dims(mean_s_1, -1)
  mean_s_2 = jnp.expand_dims(mean_s_2, -1)
  fluct_s_1 = traj_s_1 - mean_s_1
  fluct_s_2 = traj_s_2 - mean_s_2
  # calculate the (unflipped) convolution
  ft_1 = jnp.fft.rfft(fluct_s_1,
                      convolution_len)
  ft_2 = jnp.fft.rfft(fluct_s_2[..., ::-1],
                      convolution_len)
  convolved_array = jnp.fft.irfft(ft_1 * ft_2,
                                  convolution_len)[...,
                                    (frame_n - 1):]
  # normalize by the number of terms
  return(convolved_array / (jnp.arange(frame_n)[::-1] + 1))
matched_correlation_fct = jax.jit(matched_correlation_fct)


# stats based on first two cumulants
# vector/mean orientation similarity, sqrt1.sqrt2/rms1/rms2
def vector_angle_cos_s_fct(vector_s_1, vector_s_2):
  part_n = vector_s_1.shape[-1]
  return(jnp.einsum("...i, ...i -> ...", vector_s_1, vector_s_2)
         / jnp.sqrt(jnp.sum(vector_s_1 ** 2, axis = -1) 
                    * jnp.sum(vector_s_2 ** 2, axis = -1)))
vector_angle_cos_s_fct = jax.jit(vector_angle_cos_s_fct)

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

# cov orientation similarity, tr(ab)/sqrt(tr(a^2)tr(b^2))
def ori_similarity_s_fct(es_s_1, es_s_2):
  var_s_1 = es_s_1[0]
  var_s_2 = es_s_2[0]
  rot_s = jnp.swapaxes(es_s_1[1], -1, -2) @ es_s_2[1]
  return(jnp.einsum("...i, ...ij, ...ij, ...j",
                    var_s_1, rot_s, rot_s, var_s_2,
                    optimize = True)
         / jnp.sqrt(jnp.sum(var_s_1 ** 2, axis = -1) 
                    * jnp.sum(var_s_2 ** 2, axis = -1)))
ori_similarity_s_fct = jax.jit(ori_similarity_s_fct)



############################## stat_s initializers and fcts ##############################
# low res trajs
# low res trajs holder
def low_res_traj_s_initializer(condition_n_s, part_n, time_interval_s, resolution, frame_gap):
  frame_n = ((jnp.sum(jnp.round((time_interval_s[:, 1]
                                 - time_interval_s[:, 0])
                                * resolution))
              - 1) // frame_gap + 1).astype(int)
  return([jnp.zeros(tuple(condition_n_s) + (part_n, frame_n))])

# low res traj function
def low_res_traj_fct(traj, frame_gap):
  return([traj[..., ::frame_gap]])


# mean and covs
# means and covs holder
def mean_cov_s_initializer(condition_n_s, part_n):
  return([jnp.zeros(tuple(condition_n_s) + (part_n, )),
          jnp.zeros(tuple(condition_n_s) + (part_n, part_n))])

# mean cov fct
def mean_cov_s_fct(traj_s):
  mean_s = mean_s_fct(traj_s)
  return([mean_s, cov_s_fct(traj_s, mean_s)])


# pr tr os at multiple Ts
# kernel error
def kernel_error(kernel_power_s, kernel_fraction, inf, resolution):
  kernel_power_s = jnp.expand_dims(kernel_power_s, -1)
  kernel_len = 1 / kernel_fraction
  full_half_frame_n = inf * resolution
  kernel_half_frame_n = kernel_len / 2 * resolution
  return(
    jnp.sum(jnp.exp(
      -((jnp.arange(-kernel_half_frame_n,
                    kernel_half_frame_n) / resolution) ** kernel_power_s
        / (2 * 0.5 ** kernel_power_s))), axis = -1)
    / jnp.sum(jnp.exp(
      -((jnp.arange(-full_half_frame_n,
                    full_half_frame_n) / resolution) ** kernel_power_s
        / (2 * 0.5 ** kernel_power_s))), axis = -1))

# interval length needed
def min_interval_len_fct(window_len_s, samp_sep_len, samp_n,
                         max_kernel_fraction):
  if samp_sep_len == "skip_min":
    samp_sep_len = jnp.min(window_len_s)
  if samp_n == "fill_max":
    samp_n = (jnp.max(window_len_s) // samp_sep_len).astype(int)
  # max_kernel_fraction for gaussian is 0.4, for jnp.inf is 1
  min_kernel_len = jnp.ceil(jnp.max(window_len_s) / max_kernel_fraction)
  return((samp_n - 1) * samp_sep_len + min_kernel_len)

# sample separation with n
def samp_sep_with_n_fct(window_len_s, waiting_time, interval_len, 
                        max_kernel_fraction, resolution, max_samp_n = 30):
  samp_n = (jnp.max(window_len_s) // jnp.min(window_len_s)).astype(int)
  if samp_n > max_samp_n:
    samp_n = max_samp_n
  samp_sep_len = jnp.floor((interval_len - waiting_time 
                            - jnp.ceil(jnp.max(window_len_s) / max_kernel_fraction)) 
                           // samp_n)
  # peel away array
  return((int(samp_sep_len * resolution), samp_n))

# kernels
def kernel_s_fct(window_len_s, resolution, kernel_power, max_kernel_fraction):
  # create kernel, half of the window as width/std
  kernel_half_frame_n = int(jnp.ceil(jnp.max(window_len_s) / max_kernel_fraction) // 2
                            * resolution)
  if kernel_power == jnp.inf:
    kernel_s = jnp.heaviside(jnp.arange(-kernel_half_frame_n, kernel_half_frame_n)
                             + jnp.expand_dims(window_len_s / 2, 1) * resolution, 0)
    kernel_s = kernel_s * kernel_s[:, ::-1]
  else:
    kernel_s = jnp.exp(
      -((jnp.arange(-kernel_half_frame_n,
                    kernel_half_frame_n) / resolution) ** kernel_power
        / (2 * jnp.expand_dims(window_len_s / 2, 1) ** kernel_power)))
  # then normalize
  kernel_s = kernel_s / jnp.sum(kernel_s, axis = -1, keepdims = True)
  return(kernel_s)

# initialize
def multi_len_secondary_stat_s_initializer(condition_n_s, window_len_s,
                                           samp_n):
  window_len_n = window_len_s.shape[0]
  return([jnp.zeros(tuple(condition_n_s)
                    + (window_len_n + 1, samp_n))
          for stat_idx in range(5)])

# get quantities
def multi_len_secondary_stat_s_fct(traj, resolution,
                                   kernel_s, samp_sep_with_n):
  # find numbers
  # <explicit convolution prep>
  part_n = traj.shape[0]
  # </explicit convolution prep>
  # # <fourier convolution prep>
  # frame_n = traj.shape[1]
  # # </fourier convolution prep>
  kernel_frame_n = kernel_s.shape[-1]
  window_len_n = kernel_s.shape[0]
  [samp_sep, samp_n] = samp_sep_with_n
  # find reference values (window at full length)
  [full_mean, full_cov] = mean_cov_s_fct(traj)
  full_h_mean_var = jnp.var(full_mean)
  full_pc = es_s_fct(full_cov)
  full_pr = dim_r_s_fct(full_pc[0])
  full_tr = size_s_fct(full_pc[0])
  # save fluctuations and prepare for convolution (global mean used)
  fluct = traj - jnp.expand_dims(full_mean, -1)
  # <explicit convolution prep>
  temp_segmented_fluct_s = jax.lax.fori_loop(
    0, samp_n,
    lambda samp_idx, windowed_fluct_s:
    windowed_fluct_s.at[
      samp_idx].set(
        jax.lax.dynamic_slice(fluct,
                              (0, samp_idx * samp_sep),
                              (part_n, kernel_frame_n))),
    jnp.zeros((samp_n, part_n, kernel_frame_n)))
  # </explicit convolution prep>
  # # <fourier convolution prep>
  # cross_fluct_ft = jnp.fft.rfft(
  #   jnp.expand_dims(fluct, -2) * jnp.expand_dims(fluct, -3))
  # # </fourier convolution prep>
  # initialize output
  multi_len_secondary_stat_s = [
    jnp.zeros((window_len_n + 1, samp_n)).at[-1].set(full_h_mean_var),
    jnp.zeros((window_len_n + 1, samp_n)).at[-1].set(full_pr),
    jnp.zeros((window_len_n + 1, samp_n)).at[-1].set(full_tr),
    jnp.zeros((window_len_n + 1, samp_n)).at[-1].set(1),
    jnp.zeros((window_len_n + 1, samp_n)).at[-1].set(1)]
  def single_len_secondary_stat_s_updater(window_len_idx, multi_len_secondary_stat_s):
    # <explicit convolution>
    temp_cov_s = weighted_cov_s_fct(temp_segmented_fluct_s,
                                    jnp.zeros((part_n, )),
                                    kernel_s[window_len_idx])
    # </explicit convolution>
    # # <fourier convolution>
    # temp_kernel_ft = jnp.fft.rfft(kernel_s[window_len_idx], frame_n)
    # temp_cov_s = jnp.swapaxes(
    #   jnp.fft.irfft(
    #     cross_fluct_ft * temp_kernel_ft, frame_n)[
    #       ..., (kernel_frame_n - 1)::samp_sep][
    #       ..., :samp_n],
    #   -3, -1)
    # # </fourier convolution>
    temp_h_mean_var_s = (jnp.var(mean_s_fct(temp_segmented_fluct_s) + full_mean, 
                                 axis = 1))
    temp_pc_s = es_s_fct(temp_cov_s)
    temp_pr_s = dim_r_s_fct(temp_pc_s[0])
    temp_tr_s = size_s_fct(temp_pc_s[0])
    temp_os_s = ori_similarity_s_fct(full_pc, temp_pc_s)
    temp_lagged_os_s = ori_similarity_s_fct([temp_pc_s[0][0], temp_pc_s[1][0]], 
                                            temp_pc_s)
    multi_len_secondary_stat_s = [
      multi_len_secondary_stat_s[0].at[window_len_idx].set(temp_h_mean_var_s),
      multi_len_secondary_stat_s[1].at[window_len_idx].set(temp_pr_s),
      multi_len_secondary_stat_s[2].at[window_len_idx].set(temp_tr_s),
      multi_len_secondary_stat_s[3].at[window_len_idx].set(temp_os_s),
      multi_len_secondary_stat_s[4].at[window_len_idx].set(temp_lagged_os_s)]
    return(multi_len_secondary_stat_s)
  multi_len_secondary_stat_s = jax.lax.fori_loop(0, window_len_n,
                                                 single_len_secondary_stat_s_updater,
                                                 multi_len_secondary_stat_s)
  return(multi_len_secondary_stat_s)
multi_len_secondary_stat_s_fct = jax.jit(multi_len_secondary_stat_s_fct,
                                         static_argnums = (3, ))



# pr_eve_s, pr_nat_s
# # eve_s in numpy are already in columns
# rotation_eve = jnp.linalg.inv(jnp.linalg.eig(connectivity_s[connectivity_idx])[1])



############################## saving and loading ##############################
# load saved npz's as lists
def load_as_list(zipname, allow_pickle = False):
  return([jnp.load(zipname, allow_pickle = allow_pickle)[array_name] 
          for array_name in jnp.load(zipname, allow_pickle = allow_pickle).files])



############################## analysis and plotting: as scripts
