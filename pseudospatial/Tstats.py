# pr tr os at multiple Ts
# initialize
def multi_len_pr_tr_os_s_initializer(condition_n_s, interval_len_s):
  interval_len_n = interval_len_s.shape[0]
  return([jnp.zeros(tuple(condition_n_s) + (interval_len_n, ))
          for stat_idx in range(3)])

# low res traj function
def multi_len_pr_tr_os_s_fct(traj, interval_len_s):
  # find array size and initialize
  interval_len_n = interval_len_s.shape[0]
  multi_len_pr_tr_os_s_holder = [jnp.zeros((interval_len_n, )).at[-1].set(1)
                                 for stat_idx in range(3)]
  # find reference values
  long_mean = mean_s_fct(traj)
  long_cov = cov_s_fct(traj, long_mean)
  long_pc = es_s_fct(long_cov)
  long_pr = dim_r_s_fct(long_pc[0])
  long_tr = size_s_fct(long_pc[0])
  # find variables used in loop
  [part_n, step_n] = traj.shape
  short_interval_n_s = step_n // interval_len_s
  def single_len_pr_tr_os_updater(interval_len_idx, multi_len_pr_tr_os_s): # _with_ref
    traj_reshaped = jnp.swapaxes(traj.reshape(part_n,
                                              short_interval_n_s[interval_len_idx],
                                              interval_len_s[interval_len_idx]),
                                 0, 1)
    temp_short_mean_s = mean_s_fct(traj)
    temp_short_cov_s = cov_s_fct(traj, temp_short_mean_s)
    temp_short_pc_s = es_s_fct(temp_short_cov_s)
    temp_short_pr = jnp.mean(dim_r_s_fct(temp_short_pc_s[0]))
    temp_short_tr_s = size_s_fct(temp_short_pc_s[0])
    temp_short_tr = jnp.mean(temp_short_tr_s)
    temp_short_os = jnp.mean(ori_similarity_s_fct(long_pc, temp_short_pc_s,
                                                  long_tr, temp_short_tr_s))
    multi_len_pr_tr_os_s = [
      multi_len_pr_tr_os_s[0].at[interval_len_idx].set(temp_short_pr),
      multi_len_pr_tr_os_s[1].at[interval_len_idx].set(temp_short_tr),
      multi_len_pr_tr_os_s[2].at[interval_len_idx].set(temp_short_os)]
    return(multi_len_pr_tr_os_s)
  multi_len_pr_tr_os_s = jax.lax.fori_loop(0, interval_len_n - 1,
                                           single_len_pr_tr_os_updater,
                                           multi_len_pr_tr_os_s_holder)
  return(multi_len_pr_tr_os_s)

