def weighted_cov_s_fct(traj_s, mean_s, weight_s):
  # find shapes
  frame_n = traj_s.shape[-1]
  # compute
  mean_s = jnp.expand_dims(mean_s, -1)
  deviation_s = traj_s - mean_s
  cov_s = (deviation_s @ jnp.swapaxes(deviation_s * weight_s, -1, -2)
           / jnp.sum(weight_s))
  return(cov_s)
weighted_cov_s_fct = jax.jit(weighted_cov_s_fct)

# pr tr os for nTs
def multi_len_pr_tr_os_s_fct(traj, resolution,
                             window_len_s, samp_separation,
                             kernel_power = 2):
  # find numbers
  [part_n, step_n] = traj.shape
  kernel_half_step_n = step_n // 4
  window_len_n = window_len_s.shape[0]
  samp_n = (step_n - 1) // samp_separation + 1
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
  # find fluctuations first
  # local fluctuations may shift windowed covs
  fluct = traj - jnp.expand_dims(full_mean, -1)
  # initialize output
  multi_len_pr_tr_os_s = [
    jnp.zeros((window_len_n + 1, )).at[-1].set(full_pr),
    jnp.zeros((window_len_n + 1, )).at[-1].set(full_tr),
    jnp.zeros((window_len_n + 1, )).at[-1].set(1)]
  def single_len_pr_tr_os_updater(window_len_idx, multi_len_pr_tr_os_s):
    temp_segmented_fluct_s = jax.lax.fori_loop(
      0, samp_n,
      lambda samp_idx, windowed_fluct_s:
      windowed_fluct_s.at[
        samp_idx].set(
          jax.lax.dynamic_slice(traj,
                                (0, window_len_idx * samp_separation),
                                (part_n, kernel_half_step_n * 2))),
      jnp.zeros((samp_n, part_n, kernel_half_step_n * 2)))
    temp_cov_s = weighted_cov_s_fct(temp_segmented_fluct_s,
                                    jnp.zeros((part_n, )),
                                    kernel_s[window_len_idx])
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
