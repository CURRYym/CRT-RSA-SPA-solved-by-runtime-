import plotly.graph_objs as go
import scipy.io as sio
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import numpy as np

# 加载.mat文件
mat_file = sio.loadmat('trace10.mat')

# 获取变量
data = mat_file['trace']

data_list = []

for k in range(10):
    if k == 3:
        continue

    # 获取中间两段数据
    start_index1 = 1306850
    end_index1 = 2046440
    middle_data1 = data[k, start_index1:end_index1]

    # 滑动平均滤波
    window_size = 200  # 设置滑动窗口大小
    pad_width = (window_size + 1) // 2 # 边缘填充
    window = np.ones(window_size) / float(window_size)
    middle_smooth_data1 = np.convolve(middle_data1, window, 'valid')
    #middle_smooth_data2 = np.convolve(middle_data2, window, 'valid')
    # 边缘填充
    middle_smooth_data1 = np.pad(middle_smooth_data1, (pad_width, pad_width), mode='edge')
    #middle_smooth_data2 = np.pad(middle_smooth_data2, (pad_width, pad_width), mode='edge')

    # 中值滤波
    middle_medfilt_data1 = signal.medfilt(middle_smooth_data1, kernel_size = 101)
    #middle_medfilt_data2 = signal.medfilt(middle_smooth_data2, kernel_size = 101)

    # 查找峰值
    neg_data = -middle_medfilt_data1
    peaks , _ = signal.find_peaks(neg_data, distance = 600)
    neg_peaks = -neg_data[peaks]

    sep_test = np.zeros(len(middle_data1))
    for i in range(len(sep_test)):
        if i in peaks:
            sep_test[i] = -0.5

    # 统计操作类型数目
    operation_SM = len(peaks) - 1 - 512
    operation_S = 512 - operation_SM

    # 计算每个操作的运行时间
    runtime = []
    for i in range(len(peaks)-1):
        runtime.append(peaks[i+1]-160-peaks[i]-120)
    
    # 快速排序查找分割点
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        else:
            pivot = arr[0]
            left = [x for x in arr[1:] if x >= pivot]
            right = [x for x in arr[1:] if x < pivot]
            return quick_sort(left) + [pivot] + quick_sort(right)

    sorted_runtime = quick_sort(runtime)

    # 特征识别
    symbol_list = []
    feature_spiltpoint = 748
    for i in range(len(runtime)-1):
        if runtime[i] < feature_spiltpoint:
            symbol_list.append('S')
        elif runtime[i] > feature_spiltpoint-1:
            symbol_list.append('M')

    # 查找目标操作序列
    encode_str = ''.join(symbol_list)
    test_str = 'SSSMSSSSSMSSS'
    index = encode_str.find(test_str)

    if index != -1:
        print(f"索引值为 {index}")
    else:
        print(f"原字符串中不包含字符串 '{test_str}'")

    print(encode_str[index:index+91])

    data_list.append(runtime[index:index+91])

# print(data_list)

average_list = []

for i in range(91):
    sum = 0
    for j in range(9):
        sum += data_list[j][i]
    
    average_list.append(sum/9)

print(average_list)

'''
# 绘制散点图
fig = go.Figure()
fig_variance = go.Figure(data=go.Scatter(x=list(range(len(average_list))), y=average_list, mode='markers'))
fig_variance.show()
'''

# 特征识别
symbol_list = []
feature_spiltpoint = 748
for i in range(len(average_list)-1):
    if average_list[i] < feature_spiltpoint and average_list[i+1] < feature_spiltpoint:
        symbol_list.append('0')
    elif average_list[i] < feature_spiltpoint and average_list[i+1] > feature_spiltpoint-1:
        symbol_list.append('1')

symbol_str = ''.join(symbol_list)
print(symbol_str)