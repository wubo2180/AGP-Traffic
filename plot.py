import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import datetime

# import seaborn as sns
# 这两行代码解决 plt 中文显示的问题
fontsize_bar = 14
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'legend.fontsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 18,
    'font.family': 'Times New Roman'
})
#### PEMS03
# 输入统计数据
# metrics = ('MAE/RMSE/MAPE\n288', 'MAE/RMSE/MAPE\n864', 'MAE/RMSE/MAPE\n2016')
def parameter():
    metrics = ('288/864/2016\nMAE', '288/864/2016\nRMSE', '288/864/2016\nMAPE')

    # 288, 864, 2016
    # buy_number_MAE = [13.77, 13.61, 13.89 ]
    # buy_number_RMSE = [24.42, 23.91, 24.58]
    # buy_number_MAPE = [13.62, 13.10, 13.81]
    buy_number_MAE = [13.77, 24.22, 13.62 ]
    buy_number_RMSE = [13.61, 23.91, 13.10]
    buy_number_MAPE = [13.89, 24.51, 13.81]
    bar_width = 0.25  # 条形宽度
    index_mae = np.arange(len(metrics))  # 男生条形图的横坐标
    index_rmse = index_mae + bar_width  # 女生条形图的横坐标
    index_mape = index_rmse + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_mae, height=buy_number_MAE, width=bar_width, color=(1, 0.5, 0, 0.5), label='288')
    plt.bar(index_rmse, height=buy_number_RMSE, width=bar_width, color='wheat', label='864')
    plt.bar(index_mape, height=buy_number_MAPE, width=bar_width, color='darkgoldenrod', label='2016')
    for i, value in enumerate(buy_number_MAE):
        plt.text(i, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_RMSE):
        plt.text(i+0.25, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_MAPE):
        plt.text(i+0.5, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    # plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    y = range(10,28,2)
    plt.ylim((10, 28))
    plt.legend(fontsize=17)  # 显示图例
    plt.yticks(y, fontsize=17)
    plt.xticks(index_mae + 0.25, metrics, fontsize=17)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('购买量')  # 纵坐标轴标题
    plt.title('PEMS03',fontsize=20)  # 图形标题
    plt.savefig('figure/pre_k_parameter_PEMS03.pdf')
    plt.show()
    ####
    #### PEMS04
    # 输入统计数据
    # metrics = ('MAE/RMSE/MAPE\n288', 'MAE/RMSE/MAPE\n864', 'MAE/RMSE/MAPE\n2016')
    metrics = ('288/864/2016\nMAE', '288/864/2016\nRMSE', '288/864/2016\nMAPE')

    # 288, 864, 2016
    # buy_number_MAE = [13.77, 13.61, 13.89 ]
    # buy_number_RMSE = [24.42, 23.91, 24.58]
    # buy_number_MAPE = [13.62, 13.10, 13.81]
    buy_number_MAE = [17.13, 28.72, 11.62 ]
    buy_number_RMSE = [16.73, 28.39, 11.10]
    buy_number_MAPE = [17.34, 29.18, 11.77]
    bar_width = 0.25  # 条形宽度
    index_mae = np.arange(len(metrics))  # 男生条形图的横坐标
    index_rmse = index_mae + bar_width  # 女生条形图的横坐标
    index_mape = index_rmse + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_mae, height=buy_number_MAE, width=bar_width, color=(1, 0.5, 0, 0.5), label='288')
    plt.bar(index_rmse, height=buy_number_RMSE, width=bar_width, color='wheat', label='864')
    plt.bar(index_mape, height=buy_number_MAPE, width=bar_width, color='darkgoldenrod', label='2016')
    for i, value in enumerate(buy_number_MAE):
        plt.text(i, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_RMSE):
        plt.text(i+0.25, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_MAPE):
        plt.text(i+0.5, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    # plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    y = range(8,33,2)
    plt.ylim((8, 33))
    plt.legend(fontsize=17)  # 显示图例
    plt.yticks(y, fontsize=17)
    plt.xticks(index_mae + 0.25, metrics, fontsize=17)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('购买量')  # 纵坐标轴标题
    plt.title('PEMS04',fontsize=20)  # 图形标题
    plt.savefig('figure/pre_k_parameter_PEMS04.pdf')
    plt.show()

    ####
    #### PEMS07
    # 输入统计数据
    # metrics = ('MAE/RMSE/MAPE\n288', 'MAE/RMSE/MAPE\n864', 'MAE/RMSE/MAPE\n2016')
    metrics = ('288/864/2016\nMAE', '288/864/2016\nRMSE', '288/864/2016\nMAPE')

    # 288, 864, 2016
    # buy_number_MAE = [13.77, 13.61, 13.89 ]
    # buy_number_RMSE = [24.42, 23.91, 24.58]
    # buy_number_MAPE = [13.62, 13.10, 13.81]
    buy_number_MAE = [18.11, 30.91, 7.59 ]
    buy_number_RMSE = [17.78, 30.78, 7.24]
    buy_number_MAPE = [18.38, 31.15, 7.92]
    bar_width = 0.25  # 条形宽度
    index_mae = np.arange(len(metrics))  # 男生条形图的横坐标
    index_rmse = index_mae + bar_width  # 女生条形图的横坐标
    index_mape = index_rmse + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_mae, height=buy_number_MAE, width=bar_width, color=(1, 0.5, 0, 0.5), label='288')
    plt.bar(index_rmse, height=buy_number_RMSE, width=bar_width, color='wheat', label='864')
    plt.bar(index_mape, height=buy_number_MAPE, width=bar_width, color='darkgoldenrod', label='2016')
    for i, value in enumerate(buy_number_MAE):
        plt.text(i, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_RMSE):
        plt.text(i+0.25, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_MAPE):
        plt.text(i+0.5, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    # plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    y = range(4,35,2)
    plt.ylim((4, 35))
    plt.legend(fontsize=17)  # 显示图例
    plt.yticks(y, fontsize=17)
    plt.xticks(index_mae + 0.25, metrics, fontsize=17)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('购买量')  # 纵坐标轴标题
    plt.title('PEMS07',fontsize=20)  # 图形标题
    plt.savefig('figure/pre_k_parameter_PEMS07.pdf')
    plt.show()
    ####
    #### PEMS08
    # 输入统计数据
    # metrics = ('MAE/RMSE/MAPE\n288', 'MAE/RMSE/MAPE\n864', 'MAE/RMSE/MAPE\n2016')
    metrics = ('288/864/2016\nMAE', '288/864/2016\nRMSE', '288/864/2016\nMAPE')

    # 288, 864, 2016
    # buy_number_MAE = [13.77, 13.61, 13.89 ]
    # buy_number_RMSE = [24.42, 23.91, 24.58]
    # buy_number_MAPE = [13.62, 13.10, 13.81]
    buy_number_MAE = [13.32, 22.35, 8.10 ]
    buy_number_RMSE = [12.97, 22.13, 7.91]
    buy_number_MAPE = [13.49, 22.81, 8.29]
    bar_width = 0.25  # 条形宽度
    index_mae = np.arange(len(metrics))  # 男生条形图的横坐标
    index_rmse = index_mae + bar_width  # 女生条形图的横坐标
    index_mape = index_rmse + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_mae, height=buy_number_MAE, width=bar_width, color=(1, 0.5, 0, 0.5), label='288')
    plt.bar(index_rmse, height=buy_number_RMSE, width=bar_width, color='wheat', label='864')
    plt.bar(index_mape, height=buy_number_MAPE, width=bar_width, color='darkgoldenrod', label='2016')
    for i, value in enumerate(buy_number_MAE):
        plt.text(i, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_RMSE):
        plt.text(i+0.25, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_MAPE):
        plt.text(i+0.5, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    # plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    y = range(4, 26, 2)
    plt.ylim((4, 26))
    plt.legend(fontsize=17)  # 显示图例
    plt.yticks(y, fontsize=17)
    plt.xticks(index_mae + 0.25, metrics, fontsize=17)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('购买量')  # 纵坐标轴标题
    plt.title('PEMS08',fontsize=20)  # 图形标题
    plt.savefig('figure/pre_k_parameter_PEMS08.pdf')
    plt.show()

###
def mask_ratio():
    metrics = ('0.25/0.5/0.75\nMAE', '0.25/0.5/0.75\nRMSE', '0.25/0.5/0.75\nMAPE')

    # 288, 864, 2016
    # buy_number_MAE = [13.77, 13.61, 13.89 ]
    # buy_number_RMSE = [24.42, 23.91, 24.58]
    # buy_number_MAPE = [13.62, 13.10, 13.81]
    buy_number_MAE = [13.61, 23.91, 13.10]
    buy_number_RMSE = [13.89, 24.22, 13.45]
    buy_number_MAPE = [14.10, 24.39, 13.89]
    bar_width = 0.25  # 条形宽度
    index_mae = np.arange(len(metrics))  # 男生条形图的横坐标
    index_rmse = index_mae + bar_width  # 女生条形图的横坐标
    index_mape = index_rmse + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_mae, height=buy_number_MAE, width=bar_width, color='lightgreen', label='0.25')
    plt.bar(index_rmse, height=buy_number_RMSE, width=bar_width, color='limegreen', label='0.5')
    plt.bar(index_mape, height=buy_number_MAPE, width=bar_width, color='forestgreen', label='0.75')
    for i, value in enumerate(buy_number_MAE):
        plt.text(i, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_RMSE):
        plt.text(i+0.25, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_MAPE):
        plt.text(i+0.5, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    # plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    y = range(10,28,2)
    plt.ylim((10, 28))
    plt.legend(fontsize=17)  # 显示图例
    plt.yticks(y, fontsize=16)
    plt.xticks(index_mae + 0.25, metrics, fontsize=17)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('购买量')  # 纵坐标轴标题
    plt.title('PEMS03',fontsize=20)  # 图形标题
    plt.savefig('figure/pre_mask_ratio_parameter_PEMS03.pdf')
    plt.show()
    ####
    #### PEMS04
    # 输入统计数据
    # metrics = ('MAE/RMSE/MAPE\n288', 'MAE/RMSE/MAPE\n864', 'MAE/RMSE/MAPE\n2016')



    # 288, 864, 2016
    # buy_number_MAE = [13.77, 13.61, 13.89 ]
    # buy_number_RMSE = [24.42, 23.91, 24.58]
    # buy_number_MAPE = [13.62, 13.10, 13.81]
    buy_number_MAE = [16.73, 28.39, 11.10]
    buy_number_RMSE = [17.33, 28.98, 11.76]
    buy_number_MAPE = [17.47, 29.08, 11.87]
    bar_width = 0.25  # 条形宽度
    index_mae = np.arange(len(metrics))  # 男生条形图的横坐标
    index_rmse = index_mae + bar_width  # 女生条形图的横坐标
    index_mape = index_rmse + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_mae, height=buy_number_MAE, width=bar_width, color='lightgreen',label='0.25')
    plt.bar(index_rmse, height=buy_number_RMSE, width=bar_width, color='limegreen', label='0.5')
    plt.bar(index_mape, height=buy_number_MAPE, width=bar_width, color='forestgreen', label='0.75')
    for i, value in enumerate(buy_number_MAE):
        plt.text(i, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_RMSE):
        plt.text(i+0.25, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_MAPE):
        plt.text(i+0.5, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    # plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    y = range(8,33,2)
    plt.ylim((8, 33))
    plt.legend(fontsize=17)  # 显示图例
    plt.yticks(y, fontsize=16)
    plt.xticks(index_mae + 0.25, metrics, fontsize=17)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('购买量')  # 纵坐标轴标题
    plt.title('PEMS04',fontsize=20)  # 图形标题
    plt.savefig('figure/pre_mask_ratio_parameter_PEMS04.pdf')
    plt.show()

    ####
    #### PEMS07
    # 输入统计数据
    # metrics = ('MAE/RMSE/MAPE\n288', 'MAE/RMSE/MAPE\n864', 'MAE/RMSE/MAPE\n2016')

    # 288, 864, 2016
    # buy_number_MAE = [13.77, 13.61, 13.89 ]
    # buy_number_RMSE = [24.42, 23.91, 24.58]
    # buy_number_MAPE = [13.62, 13.10, 13.81]
    buy_number_MAE = [17.78, 30.78, 7.24]
    buy_number_RMSE = [18.19, 31.02, 7.52 ]
    buy_number_MAPE = [18.48, 31.27, 7.69]
    bar_width = 0.25  # 条形宽度
    index_mae = np.arange(len(metrics))  # 男生条形图的横坐标
    index_rmse = index_mae + bar_width  # 女生条形图的横坐标
    index_mape = index_rmse + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_mae, height=buy_number_MAE, width=bar_width, color='lightgreen',label='0.25')
    plt.bar(index_rmse, height=buy_number_RMSE, width=bar_width, color='limegreen', label='0.5')
    plt.bar(index_mape, height=buy_number_MAPE, width=bar_width, color='forestgreen', label='0.75')
    for i, value in enumerate(buy_number_MAE):
        plt.text(i, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_RMSE):
        plt.text(i+0.25, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_MAPE):
        plt.text(i+0.5, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    # plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    y = range(4,35,2)
    plt.ylim((4, 35))
    plt.legend(fontsize=17)  # 显示图例
    plt.yticks(y, fontsize=15)
    plt.xticks(index_mae + 0.25, metrics, fontsize=17)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('购买量')  # 纵坐标轴标题
    plt.title('PEMS07',fontsize=20)  # 图形标题
    plt.savefig('figure/pre_mask_ratio_parameter_PEMS07.pdf')
    plt.show()
    ####
    #### PEMS08
    # 输入统计数据
    # metrics = ('MAE/RMSE/MAPE\n288', 'MAE/RMSE/MAPE\n864', 'MAE/RMSE/MAPE\n2016')

    # 288, 864, 2016
    # buy_number_MAE = [13.77, 13.61, 13.89 ]
    # buy_number_RMSE = [24.42, 23.91, 24.58]
    # buy_number_MAPE = [13.62, 13.10, 13.81]
    buy_number_MAE = [12.97, 22.13, 7.91]
    buy_number_RMSE = [13.28, 22.67, 8.23]
    buy_number_MAPE = [13.39, 22.98, 8.37]
    bar_width = 0.25  # 条形宽度
    index_mae = np.arange(len(metrics))  # 男生条形图的横坐标
    index_rmse = index_mae + bar_width  # 女生条形图的横坐标
    index_mape = index_rmse + bar_width  # 女生条形图的横坐标

    # 使用两次 bar 函数画出两组条形图
    plt.bar(index_mae, height=buy_number_MAE, width=bar_width, color='lightgreen', label='0.25')
    plt.bar(index_rmse, height=buy_number_RMSE, width=bar_width, color='limegreen', label='0.5')
    plt.bar(index_mape, height=buy_number_MAPE, width=bar_width, color='forestgreen', label='0.75')
    for i, value in enumerate(buy_number_MAE):
        plt.text(i, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_RMSE):
        plt.text(i+0.25, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    for i, value in enumerate(buy_number_MAPE):
        plt.text(i+0.5, value + 0.2, str(value), ha='center', va='bottom',fontsize=fontsize_bar)
    # plt.bar(index_female, height=buy_number_female, width=bar_width, color='g', label='女性')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    y = range(4, 26, 2)
    plt.ylim((4, 26))
    plt.legend(fontsize=17)  # 显示图例
    plt.yticks(y, fontsize=16)
    plt.xticks(index_mae + 0.25, metrics, fontsize=17)  # 让横坐标轴刻度显示 waters 里的饮用水， index_male + bar_width/2 为横坐标轴刻度的位置
    # plt.ylabel('购买量')  # 纵坐标轴标题
    plt.title('PEMS08',fontsize=20)  # 图形标题
    plt.savefig('figure/pre_mask_ratio_parameter_PEMS08.pdf')
    plt.show()
def case_study():
    true_values = []
    predicted_values = [] 
# 示例代码：绘制真实值和预测值对比图
    plt.plot(true_values, label='Ground Truth')
    plt.plot(predicted_values, label='AGPST Prediction')
    plt.xlabel('Time')
    plt.ylabel('Traffic Flow')
    plt.title('Traffic Flow Prediction Comparison')
    plt.legend()
    plt.show()
def ds():


    # 假设真实数据和重建数据
    true_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.8, 2.9, 3.1, 2.2, 3.8])
    reconstructed_values = np.array([1.1, 1.9, 3.1, 3.9, 5.1, 4.7, 3.0, 3.0, 2.3, 3.6])

    # 计算误差
    errors = true_values - reconstructed_values

    # 绘制误差分布直方图
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=10, color='skyblue', edgecolor='black')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution Histogram")
    plt.grid(True)
    plt.show()


def plot_reconstruction():
    # fontsize = 24
# Load the reconstructed and original data
    loaded = np.load('./save_mask/data_16.npz')
    original_data, reconstructed_data =  loaded["array1"], loaded["array2"]
    original_data = np.squeeze(original_data)
    reconstructed_data = np.squeeze(reconstructed_data)
    reconstructed_data = reconstructed_data[:, 20]
    original_data = original_data[:, 20]

    print(original_data.shape, reconstructed_data.shape)
    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Time Series')
    plt.plot(reconstructed_data, label='Reconstructed Time Series')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Time Series Reconstruction')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig("figure/reconstruct.pdf")
    plt.show()

def plot_test_vis():
    
    # Load the reconstructed and original data
    loaded = np.load('./save_test/data_0.npz')
    reconstructed_data = np.squeeze(loaded["array1"])
    original_data = np.squeeze(loaded["array2"])

    # Slice the first 2016 time steps for a specific channel
    reconstructed_data = reconstructed_data[:2016, 0, 20]
    original_data = original_data[:2016, 0, 20]

    print(original_data.shape, reconstructed_data.shape)

    # Define tick positions for x-axis
    tick_positions_bottom = [
        "2018-11-11 00:00", 
        "2018-11-13 00:00",
        "2018-11-15 00:00",
        "2018-11-17 00:00",
        "2018-11-19 00:00"
    ]

    # Compute tick positions evenly spaced along x-axis
    tick_indices_bottom = np.linspace(0, len(original_data) - 1, len(tick_positions_bottom), dtype=int)

    # Plot the time series
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Ground Truth', linewidth=2, alpha=0.8)
    plt.plot(reconstructed_data, label='Predicted Value', linewidth=2, alpha=0.8)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title('Comparison of Ground Truth and Prediction')
    plt.ylabel('Value')

    # Set x-axis ticks correctly
    plt.xticks(tick_indices_bottom, tick_positions_bottom, rotation=0)

    plt.legend()
    plt.tight_layout()
    plt.savefig("figure/test_vis.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# Call function to plot

def plot_combined_reconstruction_and_test():
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # ===== Subplot 1: Reconstruction =====
    loaded_recon = np.load('./save_mask/data_16.npz')
    original_recon = np.squeeze(loaded_recon["array1"])[:, 20]
    recon_data = np.squeeze(loaded_recon["array2"])[:, 20]

    axes[0].plot(original_recon, label='Original Time Series', linewidth=2, alpha=0.8)
    axes[0].plot(recon_data, label='Reconstructed Time Series', linewidth=2, alpha=0.8)
    axes[0].set_title('(a) Time Series Reconstruction')
    # axes[1].text(0.5, 1.05, '(a) Time Series Reconstruction', 
    #          transform=axes[0].transAxes, 
    #           ha='center', va='top')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Value')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='upper left')

    # ===== Subplot 2: Ground Truth vs Prediction =====
    loaded_test = np.load('./save_test/data_0.npz')
    original_test = np.squeeze(loaded_test["array2"])[:2016, 0, 20]
    pred_test = np.squeeze(loaded_test["array1"])[:2016, 0, 20]

    axes[1].plot(original_test, label='Ground Truth', linewidth=2, alpha=0.8)
    axes[1].plot(pred_test, label='Predicted Value', linewidth=2, alpha=0.8)
    axes[1].set_title('(b) Comparison of Ground Truth and Prediction', loc='down')
    # axes[1].text(0.5, -0.15, '(b) Comparison of Ground Truth and Prediction', 
    #          transform=axes[1].transAxes, 
    #           ha='center', va='top')

    axes[1].set_ylabel('Value')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # Set custom x-axis ticks
    tick_positions = [
        "2018-11-11 00:00", 
        "2018-11-13 00:00",
        "2018-11-15 00:00",
        "2018-11-17 00:00",
        "2018-11-19 00:00"
    ]
    tick_indices = np.linspace(0, len(original_test) - 1, len(tick_positions), dtype=int)
    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels(tick_positions, rotation=0)
    axes[1].legend(loc='upper right')

    # Layout and save
    plt.tight_layout()
    plt.savefig("figure/combined_vis.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
font_hpyer = 24

def plot_combined_pre_k_bar_charts():
    datasets = {
        "PEMS03": {
            "MAE": [13.77, 24.22, 13.62],
            "RMSE": [13.61, 23.91, 13.10],
            "MAPE": [13.89, 24.51, 13.81],
            "ylim": (10, 28)
        },
        "PEMS04": {
            "MAE": [17.13, 28.72, 11.62],
            "RMSE": [16.73, 28.39, 11.10],
            "MAPE": [17.34, 29.18, 11.77],
            "ylim": (8, 33)
        },
        "PEMS07": {
            "MAE": [18.11, 30.91, 7.59],
            "RMSE": [17.78, 30.78, 7.24],
            "MAPE": [18.38, 31.15, 7.92],
            "ylim": (4, 35)
        },
        "PEMS08": {
            "MAE": [13.32, 22.35, 8.10],
            "RMSE": [12.97, 22.13, 7.91],
            "MAPE": [13.49, 22.81, 8.29],
            "ylim": (4, 26)
        }
    }

    metrics = ('288/864/2016\nMAE', '288/864/2016\nRMSE', '288/864/2016\nMAPE')
    bar_width = 0.25
    # fontsize_bar = 10

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (dataset, values) in enumerate(datasets.items()):
        index_mae = np.arange(len(metrics))
        index_rmse = index_mae + bar_width
        index_mape = index_rmse + bar_width

        axes[idx].bar(index_mae, values["MAE"], width=bar_width, color=(1, 0.5, 0, 0.5), label='288')
        axes[idx].bar(index_rmse, values["RMSE"], width=bar_width, color='wheat', label='864')
        axes[idx].bar(index_mape, values["MAPE"], width=bar_width, color='darkgoldenrod', label='2016')

        for i, val in enumerate(values["MAE"]):
            axes[idx].text(i, val + 0.2, str(val), ha='center', va='bottom', fontsize=fontsize_bar)
        for i, val in enumerate(values["RMSE"]):
            axes[idx].text(i + 0.25, val + 0.2, str(val), ha='center', va='bottom', fontsize=fontsize_bar)
        for i, val in enumerate(values["MAPE"]):
            axes[idx].text(i + 0.5, val + 0.2, str(val), ha='center', va='bottom', fontsize=fontsize_bar)

        axes[idx].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        axes[idx].set_ylim(values["ylim"])
        axes[idx].set_xticks(index_mae + 0.25)
        axes[idx].set_xticklabels(metrics, fontsize=font_hpyer)
        axes[idx].set_yticks(np.arange(values["ylim"][0], values["ylim"][1] + 1, 2))
        axes[idx].tick_params(axis='y', labelsize=font_hpyer)
        axes[idx].set_title(dataset, fontsize=font_hpyer)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig('figure/combined_pre_k_parameter.pdf')
    plt.show()


def plot_combined_mask_ratio_bar_charts():

    datasets = {
        'PEMS03': {
            'MAE': [13.61, 23.91, 13.10],
            'RMSE': [13.89, 24.22, 13.45],
            'MAPE': [14.10, 24.39, 13.89],
            'ylim': (10, 28),
            'yticks': range(10, 28, 2)
        },
        'PEMS04': {
            'MAE': [16.73, 28.39, 11.10],
            'RMSE': [17.33, 28.98, 11.76],
            'MAPE': [17.47, 29.08, 11.87],
            'ylim': (8, 33),
            'yticks': range(8, 33, 2)
        },
        'PEMS07': {
            'MAE': [17.78, 30.78, 7.24],
            'RMSE': [18.19, 31.02, 7.52],
            'MAPE': [18.48, 31.27, 7.69],
            'ylim': (4, 35),
            'yticks': range(4, 35, 2)
        },
        'PEMS08': {
            'MAE': [12.97, 22.13, 7.91],
            'RMSE': [13.28, 22.67, 8.23],
            'MAPE': [13.39, 22.98, 8.37],
            'ylim': (4, 26),
            'yticks': range(4, 26, 2)
        }
    }

    colors = ['lightgreen', 'limegreen', 'forestgreen']
    labels = ['0.25', '0.5', '0.75']
    metrics = ('0.25/0.5/0.75\nMAE', '0.25/0.5/0.75\nRMSE', '0.25/0.5/0.75\nMAPE')
    bar_width = 0.25

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    for ax, (title, data) in zip(axs, datasets.items()):
        index = np.arange(len(metrics))
        for i, (values, color, label) in enumerate(zip([data['MAE'], data['RMSE'], data['MAPE']], colors, labels)):
            ax.bar(index + i * bar_width, height=values, width=bar_width, color=color, label=label)
            for j, value in enumerate(values):
                ax.text(j + i * bar_width, value + 0.2, str(value), ha='center', va='bottom', fontsize=fontsize_bar)

        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(metrics, fontsize=font_hpyer)
        ax.set_yticks(data['yticks'])
        ax.set_yticklabels(data['yticks'], fontsize=font_hpyer)
        ax.set_ylim(data['ylim'])
        ax.set_title(title, fontsize=font_hpyer)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend()


    plt.tight_layout()
    plt.savefig("figure/combined_pre_mask_ratio_parameter.pdf")
    plt.show()

if __name__ == "__main__":
    plot_combined_pre_k_bar_charts()
    plot_combined_mask_ratio_bar_charts()
    # parameter()
    # mask_ratio()
    # plot_reconstruction()
    # plot_test_vis()
    # plot_combined_reconstruction_and_test()
    