#############################################
# 糖尿病预测
# 数据预处理
# 功能：读取数据 → 查看数据集形状 → 处理缺失异常值 → 清洗数据 → 保存数据
#############################################



# 查看数据集结构
# import pandas as pd

# # 1. 定义数据路径
# data_path = r"D:\diabetes\data\diabetes.csv"

# # 2. 读取CSV数据
# df = pd.read_csv(data_path)

# # 3. 查看数据集形状（行，列）
# print("\n========== 数据集结构 ==========")
# print("数据集(行数，列数):", df.shape)

# # 4. 打印前8行数据
# print("\n========== 数据集前10行 ==========")
# print(df.head(10))



# # 查看缺失值和字段类型
# import pandas as pd
# # 1. 定义数据路径
# data_path = r"D:\diabetes\data\diabetes.csv"

# # 2. 读取数据
# df = pd.read_csv(data_path)

# print("\n========== 数据字段类型与缺失值 ==========")
# print(df.info())



# # 数据统计描述

# import pandas as pd

# # 读取数据
# data_path = r"D:\diabetes\data\diabetes.csv"
# df = pd.read_csv(data_path)

# print("\n========== 数值属性统计分析 ==========")
# print(df.describe().T.round(2))



# # 绘制每个特征的箱线图判断异常0值
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # 读取数据
# df = pd.read_csv(r"D:\diabetes\data\diabetes.csv")

# # 设置风格
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # 要画的特征列表
# features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
#             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# # 创建4行2列的子图布局
# fig, axes = plt.subplots(4, 2, figsize=(12, 14))
# axes = axes.flatten()

# # 循环绘制每个特征的箱线图
# for i, col in enumerate(features):
#     ax = axes[i]
#     # 绘制箱线图，用不同颜色区分每个特征
#     sns.boxplot(y=df[col], ax=ax, color=f'C{i}')
#     ax.set_title(f'Boxplot of {col}')
#     ax.set_ylabel(col)

# # 调整布局，防止文字重叠
# plt.tight_layout()
# plt.show()



# # 各属性的直方图分布
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd

# # 读取数据
# data_path = r"D:\diabetes\data\diabetes.csv"
# df = pd.read_csv(data_path)

# # 绘制属性的直方图分布
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
# plt.rcParams['axes.unicode_minus'] = False

# # 定义要画的列和它们的标题
# columns_to_plot = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
#                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
# titles = [
#     'Distribution of Pregnancies', 'Distribution of Glucose',
#     'Distribution of BloodPressure', 'Distribution of SkinThickness',
#     'Distribution of Insulin', 'Distribution of BMI',
#     'Distribution of DiabetesPedigreeFunction', 'Distribution of Age'
# ]

# # 创建4行2列的子图布局
# fig, axes = plt.subplots(4, 2, figsize=(12, 14))
# axes = axes.flatten()  # 把二维数组转成一维，方便循环

# for i, col in enumerate(columns_to_plot):
#     ax = axes[i]
    
#     # 1. 画直方图
#     sns.histplot(
#         df[col],
#         kde=False,
#         bins=30,
#         ax=ax,
#         color='salmon',
#         edgecolor='black',
#         linewidth=1
#     )

#     # 拟合曲线
#     ax2 = ax.twinx()  # 新建一个共享X轴的Y轴
#     sns.kdeplot(
#         df[col],
#         ax=ax2,        
#         color='red',
#         linewidth=1.2
#     )
#     ax2.set_yticks([])

#     # 3. 添加统计信息文本框（均值、中位数、标准差、偏度）
#     mean = df[col].mean()
#     median = df[col].median()
#     std = df[col].std()
#     skewness = df[col].skew()
    
#     stats_text = (
#         f"Mean: {mean:.2f}\n"
#         f"Median: {median:.2f}\n"
#         f"Standard Dev: {std:.2f}\n"
#         f"Skewness: {skewness:.2f}"
#     )
    
#     # 把文本框放在右上角
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
#     ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
#             verticalalignment='top', horizontalalignment='right', bbox=props)
    
#     # 设置标题和x轴标签
#     ax.set_title(titles[i])
#     ax.set_xlabel(col)
#     ax.set_ylabel('Frequency')

# # 调整子图间距，防止重叠
# plt.tight_layout()
# plt.show()



# 处理异常0值，输出清理后的数据集
# import pandas as pd
# import numpy as np

# # 读取原始数据
# data_path = r"D:\diabetes\data\diabetes.csv"
# df = pd.read_csv(data_path)

# # 无效0值替换为NaN
# columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# for col in columns_with_zeros:
#     df[col] = df[col].replace(0, np.nan)

# # 按分布类型填充，并适配数据类型
# # 近似对称分布，用均值填充
# # Glucose：原int，均值四舍五入后转为int
# df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean()).round().astype(int)

# # SkinThickness：原int，均值四舍五入后转为int
# df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean()).round().astype(int)

# # 偏态分布，用中位数填充
# # BloodPressure：原int，中位数本身就是整数，直接转int
# df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].median()).astype(int)

# # Insulin：原int，中位数本身就是整数，直接转int
# df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median()).astype(int)

# # BMI：原float，中位数保留1位小数
# df['BMI'] = df['BMI'].fillna(df['BMI'].median()).round(1)

# # 验证数据类型和前10行
# print("填充后各列数据类型：")
# print(df.dtypes)
# print("\n填充后前10行数据：")
# print(df.head(10))

# # 保存清洗后的数据
# df.to_csv(r"D:\diabetes\data\cleaned_diabetes.csv", index=False)
# print("\n数据清洗完成，已保存为cleaned_diabetes.csv")