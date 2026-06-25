#############################################
# 糖尿病预测
# 数据分析
# 功能：
#############################################



# # 查看数据比例
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # 读取清洗后的数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 设置风格
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # 统计Outcome分布
# counts = df['Outcome'].value_counts()
# labels = ['Non-Diabetic', 'Diabetic']
# colors = ['#66b3ff', '#ff6666']  # 蓝色 = 非糖尿病，红色 = 糖尿病

# # 创建1行2列的子图
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# # 1. 水平条形图
# sns.barplot(x=counts.values, y=labels, hue=labels, palette=colors, ax=ax1, legend=False)

# ax1.set_title('Distribution of Outcome', fontsize=12)
# ax1.set_xlabel('Count')
# ax1.set_ylabel('Outcome')

# # 标注数量
# for i, v in enumerate(counts.values):
#     ax1.text(v + 8, i, str(v), color='black', fontsize=12, va='center')

# # 2. 饼图
# ax2.pie(counts.values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
# ax2.set_title('Proportion of Outcome', fontsize=12)

# plt.tight_layout()
# plt.show()



# 判定预测因子

# # 分组对比箱线图：患病 vs 未患病 特征分布差异
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # 读取清洗后的数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 中文显示设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # 所有特征列表
# features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
#             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# # 创建画布 4行2列
# fig, axes = plt.subplots(4, 2, figsize=(16, 20))
# axes = axes.flatten()

# # 颜色：非糖尿病=蓝色，糖尿病=红色
# colors = ['#66b3ff', '#ff6666']

# # 循环画每个特征
# for i, col in enumerate(features):
#     ax = axes[i]
    
#     # 按Outcome 分组画箱线图
#     sns.boxplot(x='Outcome', y=col, hue='Outcome', 
#                 palette=colors, 
#                 data=df, 
#                 ax=ax, 
#                 legend=False)
    
#     ax.set_title(f'{col} 分布对比 (0=未患病, 1=患病)', fontsize=12)
#     ax.set_xlabel('')  
    
# plt.tight_layout()
# plt.show()



# # 均值条形图：量化两组特征的平均值差异
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # 读取清洗后的数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 中文设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # 要对比的特征列表
# features = [
#     'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
# ]

# # 创建 3x3 的子图布局
# fig, axes = plt.subplots(3, 3, figsize=(16, 12))
# axes = axes.flatten()

# # 颜色
# colors = ['#66b3ff', '#ff6666']

# for i, col in enumerate(features):
#     ax = axes[i]

#     sns.barplot(
#         x='Outcome',
#         y=col,
#         hue='Outcome',
#         palette=colors,
#         data=df,
#         ax=ax,
#         legend=False,
#         errorbar=None
#     )

#     # 计算均值并标注
#     mean_val = df.groupby('Outcome')[col].mean()
#     for j, v in enumerate(mean_val):
#         ax.text(j, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)

#     ax.set_title(f'Mean {col} by Outcome', fontsize=12)
#     ax.set_ylabel(col)

# # 隐藏多余的子图
# axes[-1].axis('off')

# plt.tight_layout()
# plt.show()



# # 相关性热力图：特征与糖尿病关联强度
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # 读取清洗后的数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 计算相关系数矩阵
# corr_matrix = df.corr()

# # 设置画布大小
# plt.figure(figsize=(12, 8))

# # 绘制热力图 —— 红蓝配色，越红相关性越强
# sns.heatmap(
#     corr_matrix,
#     annot=True,        # 显示相关系数数字
#     fmt=".2f",         # 保留2位小数
#     cmap="RdBu_r",     # 红蓝配色（红=高相关，蓝=低相关）
#     linewidths=0.5,    # 格子边框
#     vmin=-1, vmax=1    # 相关性范围 -1 ~ 1
# )

# plt.title('Feature Correlation Heatmap', fontsize=14)
# plt.tight_layout()
# plt.show()



# # 医学指标分类分箱
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 读取清洗好的数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 1. 年龄分组
# df['AgeGroup'] = pd.cut(
#     df['Age'], 
#     bins=[20, 30, 40, 50, 60, 100], 
#     labels=['Young Adult', 'Early Middle Age', 'Late Middle Age', 'Early Senior', 'Senior']
# )

# # 2. BMI 分组（WHO 标准）
# df['BMI_Group'] = pd.cut(
#     df['BMI'], 
#     bins=[0, 18.5, 25, 30, 100], 
#     labels=['Underweight', 'Normal', 'Overweight', 'Obese']
# )

# # 3. 血压分组
# df['BloodPressure_Group'] = pd.cut(
#     df['BloodPressure'], 
#     bins=[0, 60, 80, 90, 120, 200], 
#     labels=['Low', 'Normal', 'Pre-hypertension', 'Stage 1 Hypertension', 'Stage 2 Hypertension']
# )

# # 4. 结果标签
# df['Outcome_Group'] = df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

# # 5. 血糖分类
# df['Glucose_Category'] = pd.cut(
#     df['Glucose'], 
#     bins=[0, 140, 200], 
#     labels=['Normal', 'Impaired glucose tolerance']
# )

# # 6. 胰岛素分类
# df['Insulin_Category'] = pd.cut(
#     df['Insulin'],
#     bins=[0, 30, 100, 150, 1000], 
#     labels=['< 30 (Possible Deficiency)', '30-100 (Normal)', '100-150 (Early Resistance)', '> 150 (Significant Resistance)'],
#     right=False
# )

# # 7. 遗传风险分组
# bins = [0, 0.5, 1.0, 1.5, df['DiabetesPedigreeFunction'].max()]
# labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '>1.5']
# df['Pedigree_Bin'] = pd.cut(
#     df['DiabetesPedigreeFunction'], 
#     bins=bins, 
#     labels=labels, 
#     right=False
# )

# # 看看分箱结果
# print("分箱完成！新增分类列：")
# print([col for col in df.columns if 'Group' in col or 'Category' in col or 'Bin' in col])



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # 读取清洗后的数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 中文设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # ---------------- 分箱----------------
# # 血糖分类
# df['Glucose_Category'] = pd.cut(
#     df['Glucose'], 
#     bins=[0, 140, 200], 
#     labels=['Normal', 'Impaired glucose tolerance']
# )
# # 结果标签
# df['Outcome_Group'] = df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

# # ---------------- 绘图：血糖分组堆叠条形图 ----------------
# # 1. 计算每个血糖类别中糖尿病/非糖尿病的数量
# ct = pd.crosstab(df['Glucose_Category'], df['Outcome_Group'])
# ct = ct[['No Diabetes', 'Diabetes']]
# # 2. 转成百分比（每个类别内的占比）
# ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

# # 绘图
# colors = ['#87CEEB', '#F08080']    # 蓝色=No Diabetes，红色=Diabetes
# ax = ct_pct.plot(
#     kind='bar',
#     stacked=True,
#     color=colors,
#     figsize=(10, 7),
#     edgecolor='black'
# )

# # 在每个块上标注百分比
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.1f%%', label_type='center')

# plt.title('Glucose Category Distribution by Outcome')
# plt.ylabel('Count')
# plt.xlabel('Glucose Category')
# plt.xticks(rotation=0)
# plt.legend(title='Outcome Group')
# plt.tight_layout()
# plt.show()



# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# # 读取清洗后的数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 中文设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # ---------------- 年龄分箱 ----------------
# df['AgeGroup'] = pd.cut(
#     df['Age'], 
#     bins=[20, 30, 40, 50, 60, 100], 
#     labels=['Young Adult', 'Early Middle Age', 'Late Middle Age', 'Early Senior', 'Senior']
# )

# # 结果标签
# df['Outcome_Group'] = df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

# # 年龄分组堆叠水平条形图
# # 1. 计算每个年龄组中糖尿病/非糖尿病的数量
# ct = pd.crosstab(df['AgeGroup'], df['Outcome_Group'])
# ct = ct[['No Diabetes', 'Diabetes']]
# # 2. 转成百分比（每个组内的占比）
# ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

# # 绘图
# colors = ['#87CEEB', '#F08080']  # 浅蓝色=No Diabetes，浅红色=Diabetes
# ax = ct_pct.plot(
#     kind='barh',  # 水平条形图
#     stacked=True,
#     color=colors,
#     figsize=(10, 7),
#     edgecolor='black'
# )

# # 在每个块上标注百分比
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.1f%%', label_type='center')

# plt.title('Percentage of Diabetes Status within Each Age Group')
# plt.xlabel('Percentage (%)')
# plt.ylabel('Age Group')
# plt.legend(title='Diabetes Status')
# plt.tight_layout()
# plt.show()



# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd

# # 读取数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 中文设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # ===================== BMI 分箱 =====================
# df['BMI_Group'] = pd.cut(
#     df['BMI'], 
#     bins=[0, 18.5, 25, 30, 100], 
#     labels=['Underweight', 'Normal', 'Overweight', 'Obese']
# )

# # 糖尿病标签
# df['Outcome_Group'] = df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

# # ===================== 绘制堆叠柱状图 =====================
# ct = pd.crosstab(df['BMI_Group'], df['Outcome_Group'])
# ct = ct[['No Diabetes', 'Diabetes']]
# ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100  # 转百分比

# # 红蓝配色
# colors = ['#87CEEB', '#F08080']

# ax = ct_pct.plot(
#     kind='bar',
#     stacked=True,
#     color=colors,
#     figsize=(10, 7),
#     edgecolor='black'
# )

# # 百分比标注
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.1f%%', label_type='center')

# plt.title('BMI Group Distribution by Outcome')
# plt.ylabel('Percentage (%)')
# plt.xlabel('BMI Group')
# plt.xticks(rotation=0)
# plt.legend(title='Outcome Group')
# plt.ylim(0, 100)
# plt.tight_layout()
# plt.show()



# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd

# # 读取数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 中文设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # ===================== 血压分箱 =====================
# df['BloodPressure_Group'] = pd.cut(
#     df['BloodPressure'],
#     bins=[0, 60, 80, 90, 120, 200],
#     labels=['Low', 'Normal', 'Pre-hypertension', 'Stage 1 Hypertension', 'Stage 2 Hypertension']
# )

# # 糖尿病标签
# df['Outcome_Group'] = df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

# # ===================== 绘制堆叠柱状图 =====================
# # 1. 计算每个血压组内糖尿病/非糖尿病的数量
# ct = pd.crosstab(df['BloodPressure_Group'], df['Outcome_Group'])
# ct = ct[['No Diabetes', 'Diabetes']]
# # 2. 转成百分比（每个组内占比，总和100%）
# ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

# # 浅蓝色=No Diabetes，浅红色=Diabetes
# colors = ['#87CEEB', '#F08080']

# ax = ct_pct.plot(
#     kind='bar',
#     stacked=True,
#     color=colors,
#     figsize=(12, 8),
#     edgecolor='black'
# )

# # 在每个块上标注百分比
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.1f%%', label_type='center')

# plt.title('Diabetes Status by Blood Pressure Group')
# plt.ylabel('Percentage (%)')
# plt.xlabel('Blood Pressure Group')
# plt.xticks(rotation=45, ha='right')  # 标签旋转45度，避免重叠
# plt.legend(title='Diabetes Status')
# plt.ylim(0, 100)
# plt.tight_layout()

# # 弹出窗口显示
# plt.show()



# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd

# # 读取数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 中文设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # ===================== 胰岛素分箱 =====================
# df['Insulin_Category'] = pd.cut(
#     df['Insulin'],
#     bins=[0, 30, 100, 150, 1000],
#     labels=['< 30 (Possible Deficiency)', 
#             '30-100 (Normal)', 
#             '100-150 (Early Resistance)', 
#             '> 150 (Significant Resistance)'],
#     right=False
# )

# df['Outcome_Group'] = df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

# ct = pd.crosstab(df['Insulin_Category'], df['Outcome_Group'])
# ct = ct[['No Diabetes', 'Diabetes']]
# ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

# # 颜色：蓝色=No Diabetes，红色=Diabetes
# colors = ['#87CEEB', '#F08080']

# ax = ct_pct.plot(
#     kind='bar',
#     stacked=True,
#     color=colors,
#     figsize=(12, 8),
#     edgecolor='black'
# )

# # 标注百分比
# for container in ax.containers:
#     ax.bar_label(container, fmt='%.1f%%', label_type='center')

# plt.title('Insulin Level Category Distribution by Outcome')
# plt.ylabel('Percentage (%)')
# plt.xlabel('Insulin Category (μU/mL)')
# plt.xticks(rotation=45, ha='right')
# plt.legend(title='Outcome Group')
# plt.ylim(0, 100)
# plt.tight_layout()

# plt.show()



# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd

# # 读取数据
# df = pd.read_csv(r"D:\diabetes\data\cleaned_diabetes.csv")

# # 中文设置
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # ===================== DPF 分箱 =====================
# bins = [0, 0.5, 1.0, 1.5, df['DiabetesPedigreeFunction'].max()]
# labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '>1.5']

# df['Pedigree_Bin'] = pd.cut(
#     df['DiabetesPedigreeFunction'],
#     bins=bins,
#     labels=labels,
#     right=False
# )

# # 糖尿病标签
# df['Outcome_Group'] = df['Outcome'].map({0: 'No Diabetes', 1: 'Diabetes'})

# # ===================== 绘制分组柱状图 =====================
# # 计算每个DPF组内糖尿病/非糖尿病的数量
# ct = pd.crosstab(df['Pedigree_Bin'], df['Outcome_Group'])
# ct = ct[['Diabetes', 'No Diabetes']]

# # 配色：红色=Diabetes，蓝色=No Diabetes
# colors = ['#F08080', '#87CEEB']

# ax = ct.plot(
#     kind='bar',
#     color=colors,
#     figsize=(12, 8),
#     edgecolor='black',
#     width=0.7
# )

# # 在每个柱子上标注数值
# for container in ax.containers:
#     ax.bar_label(container, fmt='%d', label_type='edge', padding=3)

# plt.title('Diabetes Status by Diabetes Pedigree Function Bins')
# plt.ylabel('Number of People')
# plt.xlabel('Diabetes Pedigree Function Bin')
# plt.xticks(rotation=0)
# plt.legend(title='Diabetes Status')
# plt.tight_layout()

# plt.show()



