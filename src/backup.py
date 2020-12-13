
def compute_deephit_risks_at_t_plus_delta_t(t, delta_t, preds):
    normalization = 1 - np.sum(preds[:, 0:t], axis=1).squeeze()
    risks = np.sum(preds[:, t:t + delta_t + 1], axis=1)/(normalization)
    return risks
        
def c_index_deep(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0 
    Den = 0 
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1 
        Q[i, np.where(Prediction[i] > Prediction)] = 1 
 
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1 

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

def compute_c_index_at_t_plus_delta_t(
    risks, data, start_time, time_delta
):
    
    def is_ordered_correctly( 
        risks,  first_idx, second_idx
    ):

        first_risk = risks[first_idx]
        second_risk = risks[second_idx]

        if first_risk - second_risk > 0:
            return 1
        elif first_risk == second_risk:
            # ties count as 'halfway correct'
            return 0.5
        return 0
    num_individuals = len(data.event_times)
    num_ordered_correctly = 0
    normalization = 0
    
    valid_bool_idxs_i = \
        (data.event_times <= start_time + time_delta) &\
        (~data.censoring_indicators.bool())

    valid_idxs_i = torch.arange(num_individuals)[valid_bool_idxs_i]
    
    for idx_i in valid_idxs_i:
        valid_idxs_k = torch.arange(
            num_individuals
        )[data.event_times > data.event_times[idx_i]]

        for idx_k in valid_idxs_k:
            normalization += 1
            num_ordered_correctly += is_ordered_correctly(
                risks, idx_i, idx_k
            )
    if normalization == 0:
        return 0, 0
    #print('num_ordered_correctly:', num_ordered_correctly, 'normalization:', normalization)
    c_index = num_ordered_correctly/normalization
    return c_index

if __name__ == '__main__':
    #compute non-trunc c-index
    windows = [5, 10, 15, 20]
    c_index = np.zeros([len(pred_times), len(windows)])
    c_index2 = np.zeros([len(pred_times), len(windows)])
    for p, pred_time in enumerate(pred_times):
        for w, window in enumerate(windows):
            risks = compute_deephit_risks_at_t_plus_delta_t(int(pred_time), int(window), preds_all[p])
            c_index2[p, w] = compute_c_index_at_t_plus_delta_t(risks, data, pred_time, window)
            c_index[p, w] = c_index_deep(risks, data.event_times, (~data.censoring_indicators.bool()).int(), pred_time + window)
    

    print('Average across four times regular c-index is:')
    print(np.mean(c_index, axis=1)) 
    print(np.mean(c_index2, axis=1))
