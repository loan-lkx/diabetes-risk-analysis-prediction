import os
import sys
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))
from data_processor import load_cleaned_data, FEATURES, detect_outliers_iqr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'code', 'models')
CLEANED_PATH = os.path.join(DATA_DIR, 'cleaned_diabetes.csv')

st.set_page_config(
    page_title='糖尿病分析与预测系统',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .main > div { padding: 1.5rem 2rem; }
    .st-emotion-cache-1y4p8pa { padding: 2rem 1.5rem; }
    .block-container { max-width: 1200px; padding: 2rem 1rem; }
    h1, h2, h3 { color: #1a3a5c; }
    .stMetric { background: #f0f4f8; border-radius: 12px; padding: 12px; }
    div[data-testid="stExpander"] details { border: 1px solid #e0e7ef; border-radius: 10px; }
    .pred-card { background: linear-gradient(135deg, #f0f4f8, #ffffff); border-radius: 16px; padding: 20px; border: 1px solid #e0e7ef; }
    .risk-tag { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

COLORS = px.colors.qualitative.Set2
COLOR_DIABETES = '#ef553b'
COLOR_NODIABETES = '#636efa'

FEATURE_LABELS_CN = {
    'Pregnancies': '怀孕次数',
    'Glucose': '血糖 (mg/dL)',
    'BloodPressure': '血压 (mmHg)',
    'SkinThickness': '皮肤厚度 (mm)',
    'Insulin': '胰岛素 (μU/mL)',
    'BMI': 'BMI',
    'DiabetesPedigreeFunction': '遗传函数 (DPF)',
    'Age': '年龄'
}

MODEL_NAMES_CN = {
    'KNN': 'KNN (K近邻)',
    'Logistic Regression': '逻辑回归',
    'Decision Tree': '决策树',
    'Random Forest': '随机森林',
    'Tuned Random Forest': '调优随机森林',
}


@st.cache_data
def load_data():
    return load_cleaned_data(CLEANED_PATH)


@st.cache_resource
def load_models():
    models = {}
    model_names = {
        'knn': 'KNN', 'logistic': 'Logistic Regression',
        'decision_tree': 'Decision Tree', 'random_forest': 'Random Forest',
        'tuned_random_forest': 'Tuned Random Forest',
    }
    for key, display_name in model_names.items():
        path = os.path.join(MODEL_DIR, f'{key}.pkl')
        if os.path.exists(path):
            models[display_name] = joblib.load(path)
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return models, scaler


def format_feature_name(en):
    return FEATURE_LABELS_CN.get(en, en)


def build_metric_card(label, value, color='#1a3a5c'):
    return f"""
    <div style="background:white;border-radius:12px;padding:16px 20px;border:1px solid #e0e7ef;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);text-align:center;">
        <div style="font-size:0.85rem;color:#6b7a8f;margin-bottom:6px;">{label}</div>
        <div style="font-size:1.8rem;font-weight:700;color:{color};">{value}</div>
    </div>
    """


cleaned_df = load_data()
models_dict, scaler = load_models()

# ─── 侧边栏 ───
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0;">
        <span style="font-size:2rem;">🏥</span>
        <h3 style="margin:4px 0 0;color:#1a3a5c;">糖尿病分析系统</h3>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio('导航', ['📊 数据探索与分析', '🔬 患病风险预测'], label_visibility='collapsed')
    st.divider()
    st.markdown(f'**样本数** {len(cleaned_df)} &nbsp;&nbsp; **特征数** 8')
    st.markdown(f'**患病** {(cleaned_df["Outcome"]==1).sum()} &nbsp;&nbsp; **未患病** {(cleaned_df["Outcome"]==0).sum()}')
    pos_rate = cleaned_df['Outcome'].mean()
    st.progress(pos_rate, text=f'患病率 {pos_rate:.1%}')
    st.divider()
    st.caption('基于 Pima Indians Diabetes 数据集')

# ═══════════════════════════════════════════════════
# 页面1：数据探索与分析
# ═══════════════════════════════════════════════════
if page == '📊 数据探索与分析':
    st.title('📊 数据探索与分析')
    tab1, tab2, tab3, tab4 = st.tabs([
        '📋 数据概览', '📈 特征分布',
        '🔗 相关性分析', '🤖 模型评估'
    ])

    # ── TAB 1: 数据概览 ──
    with tab1:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.subheader('清洗后数据预览')
            st.dataframe(cleaned_df.head(8), width='stretch', height=260)
        with col2:
            st.subheader('统计描述')
            desc = cleaned_df.describe().T.round(2)
            desc.columns = ['样本数', '均值', '标准差', '最小值', '25%', '50%', '75%', '最大值']
            desc.index = [format_feature_name(i) if i != 'Outcome' else '结果' for i in desc.index]
            st.dataframe(desc, width='stretch', height=260)

        col3, col4 = st.columns([1.2, 1])
        with col3:
            st.subheader('结果分布')
            counts = cleaned_df['Outcome'].value_counts()
            fig = go.Figure()
            fig.add_trace(go.Pie(
                labels=['未患病', '患病'],
                values=counts.values,
                marker=dict(colors=[COLOR_NODIABETES, COLOR_DIABETES], line=dict(color='white', width=2)),
                textinfo='label+percent',
                hole=0.45,
                showlegend=False,
            ))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            st.subheader('数据清洗说明')
            st.info(
                '**异常值处理**\n\n'
                'Glucose、BloodPressure、SkinThickness、Insulin、BMI '
                '中的 0 值被视为无效缺失值，已替换处理：\n\n'
                '- Glucose、SkinThickness → **均值填充**\n'
                '- BloodPressure、Insulin → **中位数填充**\n'
                '- BMI → **中位数填充**\n\n'
                f'清洗后数据量：**{len(cleaned_df)}** 条'
            )

    # ── TAB 2: 特征分布 ──
    with tab2:
        st.subheader('异常值检测（IQR 方法）')
        outlier_results = detect_outliers_iqr(cleaned_df)
        outlier_df = pd.DataFrame([
            {'特征': format_feature_name(f),
             '下界': f'{info["lower"]:.2f}',
             '上界': f'{info["upper"]:.2f}',
             '异常值数': info['count'],
             '异常占比': f'{info["percent"]}%'}
            for f, info in outlier_results.items()
        ])
        st.dataframe(outlier_df, width='stretch', hide_index=True)

        st.divider()
        st.subheader('特征分布直方图')
        feat_grid = [FEATURES[i:i + 4] for i in range(0, len(FEATURES), 4)]
        for row in feat_grid:
            cols = st.columns(4)
            for ci, feat in enumerate(row):
                with cols[ci]:
                    mean_v = cleaned_df[feat].mean()
                    median_v = cleaned_df[feat].median()
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=cleaned_df[feat], nbinsx=30,
                        marker=dict(color='#636efa', line=dict(color='white', width=0.8)),
                        name='',
                        hovertemplate='区间: %[x]<br>频数: %[y]<extra></extra>',
                    ))
                    fig.add_vline(x=mean_v, line=dict(color='#ef553b', width=2, dash='dash'),
                                  annotation_text=f'均值 {mean_v:.1f}', annotation_position='top')
                    fig.add_vline(x=median_v, line=dict(color='#00cc96', width=2, dash='dot'),
                                  annotation_text=f'中位数 {median_v:.1f}', annotation_position='top right')
                    fig.update_layout(
                        title=dict(text=format_feature_name(feat), font=dict(size=13)),
                        height=240, margin=dict(l=10, r=10, t=40, b=20),
                        xaxis_title=None, yaxis_title='频数',
                        showlegend=False, bargap=0.05,
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader('箱线图（特征分布与异常值）')
        box_grid = [FEATURES[i:i + 4] for i in range(0, len(FEATURES), 4)]
        for row in box_grid:
            cols = st.columns(4)
            for ci, feat in enumerate(row):
                with cols[ci]:
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=cleaned_df[feat], name='',
                        marker_color='#636efa',
                        boxmean='sd',
                        hovertemplate='值: %[y]<extra></extra>',
                    ))
                    fig.update_layout(
                        title=dict(text=format_feature_name(feat), font=dict(size=12)),
                        height=280, margin=dict(l=10, r=10, t=40, b=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: 相关性分析 ──
    with tab3:
        corr = cleaned_df.corr()

        col1, col2 = st.columns([1, 1.1])
        with col1:
            st.subheader('特征相关性热力图')
            labels_cn = [format_feature_name(c) if c != 'Outcome' else '结果' for c in corr.columns]
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=labels_cn, y=labels_cn,
                colorscale='RdBu_r', zmin=-1, zmax=1,
                text=corr.values.round(2), texttemplate='%{text}',
                hovertemplate='%{x} ↔ %{y}: %{z:.3f}<extra></extra>',
            ))
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=60))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader('与结果的相关性排序')
            corr_target = cleaned_df.corr()['Outcome'].drop('Outcome').sort_values(ascending=False)
            colors_bar = [COLOR_DIABETES if v > 0 else COLOR_NODIABETES for v in corr_target.values]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=corr_target.values, y=[format_feature_name(i) for i in corr_target.index],
                orientation='h', marker=dict(color=colors_bar, line=dict(color='white', width=1)),
                text=corr_target.values.round(3), textposition='outside',
                hovertemplate='%{y}: %{x:.3f}<extra></extra>',
            ))
            fig.update_layout(
                height=360, margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(range=[-0.2, 0.5]), yaxis=dict(autorange='reversed'),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader('分组患病率分析')
        df_display = cleaned_df.copy()
        df_display['结果'] = df_display['Outcome'].map({0: '未患病', 1: '患病'})

        def plot_group_pct(df, col, bins, labels, title, xlabel):
            df['_group'] = pd.cut(df[col], bins=bins, labels=labels)
            ct = pd.crosstab(df['_group'], df['结果'])
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            fig = go.Figure()
            for idx, label in enumerate(['未患病', '患病']):
                color = COLOR_NODIABETES if label == '未患病' else COLOR_DIABETES
                fig.add_trace(go.Bar(
                    name=label, x=ct_pct.index, y=ct_pct[label],
                    marker_color=color, text=ct_pct[label].round(1),
                    texttemplate='%{text}%', textposition='inside',
                    hovertemplate='%{x}<br>%{label}: %{y:.1f}%<extra></extra>',
                ))
            fig.update_layout(
                barmode='stack', title=title, height=300,
                xaxis_title=xlabel, yaxis_title='百分比 (%)',
                margin=dict(l=10, r=10, t=40, b=30),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            )
            return fig

        group_tabs = st.tabs(['血糖', 'BMI', '年龄', '血压', '胰岛素', '遗传函数'])
        with group_tabs[0]:
            fig = plot_group_pct(df_display, 'Glucose', [0, 100, 125, 200],
                                 ['正常 (<100)', '前期 (100-125)', '糖尿病 (>125)'],
                                 '不同血糖水平的患病率', '血糖 (mg/dL)')
            st.plotly_chart(fig, use_container_width=True)
        with group_tabs[1]:
            fig = plot_group_pct(df_display, 'BMI', [0, 18.5, 25, 30, 100],
                                 ['偏瘦', '正常', '超重', '肥胖'],
                                 '不同BMI分组的患病率', 'BMI')
            st.plotly_chart(fig, use_container_width=True)
        with group_tabs[2]:
            fig = plot_group_pct(df_display, 'Age', [20, 30, 40, 50, 60, 100],
                                 ['20-30岁', '30-40岁', '40-50岁', '50-60岁', '60岁以上'],
                                 '不同年龄段的患病率', '年龄')
            st.plotly_chart(fig, use_container_width=True)
        with group_tabs[3]:
            fig = plot_group_pct(df_display, 'BloodPressure', [0, 60, 80, 90, 120, 200],
                                 ['偏低', '正常', '高血压前期', '1级高血压', '2级高血压'],
                                 '不同血压水平的患病率', '血压 (mmHg)')
            st.plotly_chart(fig, use_container_width=True)
        with group_tabs[4]:
            fig = plot_group_pct(df_display, 'Insulin', [0, 30, 100, 150, 1000],
                                 ['缺乏 (<30)', '正常 (30-100)', '早期抵抗 (100-150)', '显著抵抗 (>150)'],
                                 '不同胰岛素水平的患病率', '胰岛素 (μU/mL)')
            st.plotly_chart(fig, use_container_width=True)
        with group_tabs[5]:
            bins = [0, 0.5, 1.0, 1.5, df_display['DiabetesPedigreeFunction'].max()]
            labels = ['0-0.5', '0.5-1.0', '1.0-1.5', '>1.5']
            fig = plot_group_pct(df_display, 'DiabetesPedigreeFunction', bins, labels,
                                 '不同遗传风险分组的患病率', '遗传函数 (DPF)')
            st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4: 模型评估 ──
    with tab4:
        st.subheader('分类模型性能对比')
        with st.spinner('正在评估模型...'):
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            from sklearn.preprocessing import StandardScaler

            X = cleaned_df.drop('Outcome', axis=1)
            y = cleaned_df['Outcome']
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            scaler_local = StandardScaler()
            X_tr_s = scaler_local.fit_transform(X_tr)
            X_te_s = scaler_local.transform(X_te)

            model_keys = ['KNN', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Tuned Random Forest']
            model_objs = {}
            for k in model_keys:
                path = os.path.join(MODEL_DIR, f'{k.lower().replace(" ","_").replace("tuned_","tuned_random_")}.pkl')
                if not os.path.exists(path):
                    path = os.path.join(MODEL_DIR, f'{k.lower().replace(" ","_").replace("tuned_","tuned_random_")}.pkl')
                # handle special cases
                if k == 'Tuned Random Forest':
                    path = os.path.join(MODEL_DIR, 'tuned_random_forest.pkl')
                elif k == 'Logistic Regression':
                    path = os.path.join(MODEL_DIR, 'logistic.pkl')
                elif k == 'Random Forest':
                    path = os.path.join(MODEL_DIR, 'random_forest.pkl')
                elif k == 'Decision Tree':
                    path = os.path.join(MODEL_DIR, 'decision_tree.pkl')
                elif k == 'KNN':
                    path = os.path.join(MODEL_DIR, 'knn.pkl')
                model_objs[k] = joblib.load(path)
                model_objs[k].fit(X_tr_s, y_tr)

            metrics_list = []
            roc_traces = []

            roc_traces.append(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(color='gray', width=1.5, dash='dash'),
                name='随机分类器 (AUC=0.500)'
            ))

            for name, model in model_objs.items():
                y_pred = model.predict(X_te_s)
                y_prob = model.predict_proba(X_te_s)[:, 1]
                acc = accuracy_score(y_te, y_pred)
                prec = precision_score(y_te, y_pred)
                rec = recall_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred)
                auc_val = roc_auc_score(y_te, y_prob)
                metrics_list.append({
                    '模型': MODEL_NAMES_CN.get(name, name),
                    '准确率': f'{acc:.4f}', '精确率': f'{prec:.4f}',
                    '召回率': f'{rec:.4f}', 'F1分数': f'{f1:.4f}', 'AUC': f'{auc_val:.4f}',
                })
                fpr, tpr, _ = roc_curve(y_te, y_prob)
                roc_traces.append(go.Scatter(
                    x=fpr, y=tpr, mode='lines',
                    line=dict(width=2.5),
                    name=f'{MODEL_NAMES_CN.get(name, name)} (AUC={auc_val:.3f})',
                    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
                ))

            fig_roc = go.Figure(data=roc_traces)
            fig_roc.update_layout(
                title='ROC 曲线对比', height=460,
                xaxis=dict(title='假阳性率 (FPR)', range=[0, 1]),
                yaxis=dict(title='真阳性率 (TPR)', range=[0, 1]),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1, font=dict(size=10)),
                margin=dict(l=10, r=10, t=50, b=30),
                hovermode='x unified',
            )

        col1, col2 = st.columns([1.3, 1])
        with col1:
            st.plotly_chart(fig_roc, use_container_width=True)
        with col2:
            st.dataframe(pd.DataFrame(metrics_list).set_index('模型'), width='stretch')

        st.divider()
        st.subheader('混淆矩阵')

        cm_grid_rows = 2
        cm_grid_cols = 3
        fig_cm = make_subplots(
            rows=cm_grid_rows, cols=cm_grid_cols,
            subplot_titles=[MODEL_NAMES_CN.get(n, n) for n in model_objs.keys()],
            horizontal_spacing=0.06, vertical_spacing=0.15,
        )
        for idx, (name, model) in enumerate(model_objs.items()):
            y_pred = model.predict(X_te_s)
            cm = confusion_matrix(y_te, y_pred)
            row = idx // cm_grid_cols + 1
            col = idx % cm_grid_cols + 1
            fig_cm.add_trace(go.Heatmap(
                z=cm, x=['未患病', '患病'], y=['未患病', '患病'],
                text=cm, texttemplate='%{text}', colorscale='Blues',
                hovertemplate='真实: %{y}<br>预测: %{x}<br>数量: %{z}<extra></extra>',
                showscale=False,
            ), row=row, col=col)
        fig_cm.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

# ═══════════════════════════════════════════════════
# 页面2：患病风险预测
# ═══════════════════════════════════════════════════
else:
    st.title('🔬 糖尿病患病风险预测')
    st.markdown(
        '<div style="background:linear-gradient(135deg,#eef2f7,#fff);border-radius:12px;padding:14px 20px;'
        'margin-bottom:20px;border:1px solid #e0e7ef;color:#2c3e50;">'
        '请输入患者健康指标并选择模型，点击预测获取诊断结果与风险评估</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap='medium')

    with col1:
        st.markdown('##### 基础生理指标')
        pregnancies = st.number_input('怀孕次数', min_value=0, max_value=20, value=2, step=1,
                                      help='怀孕次数')
        glucose = st.number_input('血糖 (mg/dL)', min_value=0, max_value=300, value=120, step=1,
                                  help='血浆葡萄糖浓度')
        blood_pressure = st.number_input('舒张压 (mmHg)', min_value=0, max_value=200, value=70, step=1,
                                         help='舒张压')
        skin_thickness = st.number_input('皮褶厚度 (mm)', min_value=0, max_value=100, value=20, step=1,
                                         help='肱三头肌皮褶厚度')
        insulin = st.number_input('血清胰岛素 (μU/mL)', min_value=0, max_value=1000, value=80, step=1,
                                  help='2小时血清胰岛素')

    with col2:
        st.markdown('##### 代谢与遗传指标')
        bmi = st.number_input('BMI 身体质量指数', min_value=0.0, max_value=80.0, value=28.0, step=0.1,
                              help='体重(kg) / 身高(m)²')
        dpf = st.number_input('糖尿病遗传函数 (DPF)', min_value=0.0, max_value=3.0, value=0.5, step=0.001,
                              format='%.3f', help='糖尿病家族遗传风险系数')
        age = st.number_input('年龄', min_value=18, max_value=120, value=35, step=1, help='年龄（岁）')

        st.markdown('<br>', unsafe_allow_html=True)
        model_options = list(models_dict.keys())
        if not model_options:
            st.error('未找到训练好的模型！请先运行 train_models.py')
            st.stop()

        selected_model = st.selectbox(
            '选择分类模型',
            model_options,
            format_func=lambda x: MODEL_NAMES_CN.get(x, x),
        )
        predict_clicked = st.button('🔍 预测糖尿病风险', type='primary', use_container_width=True)

    if predict_clicked:
        input_data = np.array([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data) if scaler is not None else input_data

        model = models_dict[selected_model]
        proba = model.predict_proba(input_scaled)[0]
        prediction = model.predict(input_scaled)[0]
        prob_no = proba[0]
        prob_yes = proba[1]

        st.divider()
        st.markdown(f'<h3 style="color:#1a3a5c;">预测结果 <span style="font-size:0.9rem;font-weight:400;color:#6b7a8f;">(模型: {MODEL_NAMES_CN.get(selected_model, selected_model)})</span></h3>',
                    unsafe_allow_html=True)

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(build_metric_card('未患病概率', f'{prob_no:.1%}', COLOR_NODIABETES), unsafe_allow_html=True)
        with r2:
            st.markdown(build_metric_card('患病概率', f'{prob_yes:.1%}', COLOR_DIABETES), unsafe_allow_html=True)
        with r3:
            if prediction == 0:
                st.markdown(build_metric_card('诊断结论', '✅ 未患病', '#00a65a'), unsafe_allow_html=True)
            else:
                st.markdown(build_metric_card('诊断结论', '⚠️ 患病', '#d9534f'), unsafe_allow_html=True)
        with r4:
            risk_level = '低风险' if prob_yes < 0.3 else ('中风险' if prob_yes < 0.6 else '高风险')
            risk_color = '#00a65a' if risk_level == '低风险' else ('#f39c12' if risk_level == '中风险' else '#d9534f')
            st.markdown(build_metric_card('风险评估', risk_level, risk_color), unsafe_allow_html=True)

        st.divider()
        st.subheader('风险因素分析')

        warnings = []
        if glucose > 125:
            warnings.append(('🔴 高血糖', f'{glucose} mg/dL（超过125，已达糖尿病范围）'))
        elif glucose > 100:
            warnings.append(('🟡 血糖偏高', f'{glucose} mg/dL（100-125，糖尿病前期）'))

        if bmi >= 30:
            warnings.append(('🔴 肥胖', f'BMI {bmi}（≥30，肥胖）'))
        elif bmi >= 25:
            warnings.append(('🟡 超重', f'BMI {bmi}（25-30，超重）'))

        if blood_pressure > 80:
            warnings.append(('🔴 血压偏高', f'{blood_pressure} mmHg（>80）'))

        if age >= 45:
            warnings.append(('🟡 年龄因素', f'{age}岁（≥45，风险升高）'))

        if dpf > 1.0:
            warnings.append(('🔴 遗传风险高', f'DPF {dpf:.3f}（>1.0）'))

        if warnings:
            warn_cols = st.columns(2)
            for wi, (icon, detail) in enumerate(warnings):
                with warn_cols[wi % 2]:
                    st.warning(f'**{icon}** — {detail}')
        else:
            st.info('✅ 基于常见医学阈值，未检测到显著风险因素。')

        st.divider()
        st.subheader('输入值与数据集均值对比')

        avg_stats = cleaned_df.describe().T[['mean', 'std']]
        comp_data = []
        for feat in FEATURES:
            user_val = {'Pregnancies': pregnancies, 'Glucose': glucose,
                        'BloodPressure': blood_pressure, 'SkinThickness': skin_thickness,
                        'Insulin': insulin, 'BMI': bmi,
                        'DiabetesPedigreeFunction': dpf, 'Age': age}[feat]
            mean_val = avg_stats.loc[feat, 'mean']
            std_val = avg_stats.loc[feat, 'std']
            diff = user_val - mean_val
            comp_data.append({
                '指标': format_feature_name(feat),
                '您的值': f'{user_val:.1f}',
                '数据集均值': f'{mean_val:.2f}',
                '差值': f'{diff:+.2f}',
                '偏离程度': '偏高 ↑' if diff > std_val else ('偏低 ↓' if diff < -std_val else '正常 →'),
            })
        st.dataframe(pd.DataFrame(comp_data), width='stretch', hide_index=True)
