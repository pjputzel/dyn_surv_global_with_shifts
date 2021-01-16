from data_handling.DataInput import DataInput
from utils.ParameterParser import ParameterParser
import torch
import numpy as np

if __name__ == "__main__":
    path_to_config = '../configs/dummy_global.yaml'
    params = ParameterParser(path_to_config).parse_params()
    torch.random.manual_seed(params['random_seed'])
    np.random.seed(params['random_seed'])
    torch.set_default_dtype(torch.float64)

    data_input = DataInput(params['data_input_params'])
    data_input.load_data()
    
    complete_times = []
    for i, times in enumerate(data_input.cov_times):
        #ignore the padding
        valid_times = torch.cat([torch.tensor([0.]), times[(times > 0)]])
        for t, time in enumerate(valid_times):
            complete_times.append([i, time.item() * 365, 0])
        print(time * 365, data_input.event_times[i] * 365)
        assert(time < data_input.event_times[i])
        if data_input.censoring_indicators[i]:
            complete_times.append([i, data_input.event_times[i].item() * 365, 0])
        else:
            complete_times.append([i, data_input.event_times[i].item() * 365, 1])

    complete_times = np.array(complete_times)
    np.savetxt('timestamp_data.csv', complete_times, delimiter=',', fmt=['%d', '%.2f', '%d'])
                
