# variables:

# pr tr os at multiple Ts
# find sampling frequency
def samp_separation_fct(interval_len, resolution,
                        window_len_s, frame_n_threshold):
  min_window_len = jnp.min(window_len_s)
  min_frame_n = (min_window_len * resolution)
  if min_frame_n > frame_n_threshold:
    step_n = interval_len * resolution
    samp_n = (interval_len / 2 / min_window_len + 1
              # have an additional sample for more samples?
              + 1)
    output = jnp.floor(step_n / samp_n).astype(int)
  else:
    print(
      "frame number ({0:.1f}) lower than required ({1:})".format(
        min_frame_n, frame_n_threshold))  
  return(output)

# initialize
def multi_len_pr_tr_os_s_initializer(condition_n_s, window_len_s):
  window_len_n = window_len_s.shape[0]
  return([jnp.zeros(tuple(condition_n_s) + (window_len_n + 1, ))
          for stat_idx in range(3)])

# pr tr os for nTs
def multi_len_pr_tr_os_s_fct(traj, resolution,
                             window_len_s, samp_separation,
                             kernel_power = 2):
  # find numbers
  step_n = traj.shape[-1]
  kernel_half_step_n = step_n // 4
  window_len_n = window_len_s.shape[0]
  # half of the window as the kernel std, then normalize
  kernel_s = jnp.exp(
    -((jnp.arange(-kernel_half_step_n,
                  kernel_half_step_n) / resolution) ** kernel_power
      / (2 * jnp.expand_dims(window_len_s / 2, 1) ** kernel_power)))
  kernel_s = kernel_s / jnp.sum(kernel_s, axis = -1, keepdims = True)
  # find reference values (window at full length)
  full_mean = mean_s_fct(traj)
  full_cov = cov_s_fct(traj, full_mean)
  full_pc = es_s_fct(full_cov)
  full_pr = dim_r_s_fct(full_pc[0])
  full_tr = size_s_fct(full_pc[0])
  # find Fourier transforms of cross fluctuations first
  # local fluctuations may shift windowed covs
  fluct = traj - jnp.expand_dims(full_mean, -1)
  cross_fluct_ft = jnp.fft.rfft(
    jnp.expand_dims(fluct, -2) * jnp.expand_dims(fluct, -3))
  # initialize output
  multi_len_pr_tr_os_s = [
    jnp.zeros((window_len_n + 1, )).at[-1].set(full_pr),
    jnp.zeros((window_len_n + 1, )).at[-1].set(full_tr),
    jnp.zeros((window_len_n + 1, )).at[-1].set(1)]
  def single_len_pr_tr_os_updater(window_len_idx, multi_len_pr_tr_os_s):
    temp_kernel_ft = jnp.fft.rfft(kernel_s[window_len_idx], step_n)
    temp_cov_s = jnp.swapaxes(
      jnp.fft.irfft(
        cross_fluct_ft * temp_kernel_ft)[
          ..., (2 * kernel_half_step_n - 1)::samp_separation],
      -3, -1)
    temp_pc_s = es_s_fct(temp_cov_s)
    temp_pr = jnp.mean(dim_r_s_fct(temp_pc_s[0]))
    temp_tr_s = size_s_fct(temp_pc_s[0])
    temp_tr = jnp.mean(temp_tr_s)
    temp_os = jnp.mean(ori_similarity_s_fct(full_pc, temp_pc_s,
                                            full_tr, temp_tr_s))
    multi_len_pr_tr_os_s = [
      multi_len_pr_tr_os_s[0].at[window_len_idx].set(temp_pr),
      multi_len_pr_tr_os_s[1].at[window_len_idx].set(temp_tr),
      multi_len_pr_tr_os_s[2].at[window_len_idx].set(temp_os)]
    return(multi_len_pr_tr_os_s)
  multi_len_pr_tr_os_s = jax.lax.fori_loop(0, window_len_n,
                                           single_len_pr_tr_os_updater,
                                           multi_len_pr_tr_os_s)
  return(multi_len_pr_tr_os_s)
multi_len_pr_tr_os_s_fct = jax.jit(multi_len_pr_tr_os_s_fct, static_argnums = (3, ))
