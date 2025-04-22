'''
Created on April 21, 2025

@author: immanueltrummer and Markus Buckwalter
'''
import json
import os
import pandas as pd
import pathlib
import psycopg2.extras
import stable_baselines3
import streamlit as st
import sys
import time

cur_file_dir = os.path.dirname(__file__)
src_dir = pathlib.Path(cur_file_dir).parent.parent
sys.path.append(str(src_dir))

import nminer.algs.rl
import nminer.sql.pred
import nminer.text.sum

st.set_page_config(page_title='NaturalMiner')

# --- SIDEBAR: Scenario & DB ---
# Removed sidebar image (icon)
st.sidebar.title("NaturalMiner")
st.sidebar.markdown("Mine data for patterns in natural language.")
st.sidebar.divider()

# Use correct path for scenarios.json (demo/ at project root)
import pathlib
project_root = pathlib.Path(__file__).parent.parent.parent.parent
scenarios_path = project_root.joinpath('demo').joinpath('scenarios.json')
with open(scenarios_path) as file:
    scenarios = json.load(file)
nr_scenarios = len(scenarios)

selected = st.sidebar.selectbox(
    'Scenario', options=range(nr_scenarios), 
    format_func=lambda idx:scenarios[idx]['scenario'])
st.sidebar.markdown("Database Connection")
connection_info = st.sidebar.text_input(
    'DB (format: <Database>:<User>:<Password>)',
    value=scenarios[selected]['dbconnection'],
    help='Example: picker:ubuntu:'
).split(':')
db_name = connection_info[0]
db_user = connection_info[1]
db_pwd = connection_info[2] if len(connection_info) > 2 else ''
table = st.sidebar.text_input(
    'Table name', max_chars=100,
    value=scenarios[selected]['table'],
    help='Name of the table to analyze.'
)
st.sidebar.divider()

# --- MAIN PAGE ---
st.markdown("""
<style>
.big-title {font-size:2.2em;font-weight:700;margin-bottom:0.2em;}
.section-header {font-size:1.3em;font-weight:600;margin-top:1.5em;color:#0068c9;}
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="big-title">Pattern Mining Assistant</div>', unsafe_allow_html=True)
st.info("""**How it works:**\nDescribe the pattern you want to find. Adjust options if needed. Click **Find Pattern!** to mine your data.""")

st.markdown('<div class="section-header">Describe the Pattern</div>', unsafe_allow_html=True)
label = st.text_input(
    'Pattern or Fact (plain English)', max_chars=100, 
    value=scenarios[selected]['goal'],
    help='E.g., "it\'s a great laptop".')

col1, col2, col3 = st.columns(3)
with col1:
    nr_facts = st.number_input('Facts to find', min_value=1, max_value=3, value=scenarios[selected]['nr_facts'], step=1, help='How many facts to extract.')
with col2:
    nr_preds = st.number_input('Predicates per fact', min_value=0, max_value=3, value=scenarios[selected]['nr_preds'], step=1, help='How many conditions per fact.')
with col3:
    nr_iterations = st.number_input('Mining iterations', min_value=1, max_value=500, value=200, step=10, help='Higher = more thorough, slower.')

with st.expander('Advanced: Text Templates & Predicates'):
    st.caption('Control how facts/aggregates are phrased and which data subsets are analyzed.')
    preamble = st.text_input(
        'Fact preamble', 
        value=scenarios[selected]['preamble'],
        help='Text that starts each fact, e.g., "Among all laptops".'
    )
    dims_info = st.text_area(
        'Dimension columns (one per line, <Column>:<TemplateText>)', 
        value=scenarios[selected]['dimensions'], height=8,
        help='Format: <Column>:<TemplateText>. Use <V> as placeholder for values.'
    ).split('\n')
    aggs_info = st.text_area(
        'Aggregation columns (one per line, <Column>:<TemplateText>)', 
        value=scenarios[selected]['aggregates'], height=8,
        help='Format: <Column>:<TemplateText>.'
    ).split('\n')
    cmp_preds = st.text_area(
        'Data subsets (one SQL predicate per line)',
        value=scenarios[selected]['predicates'], height=8,
        help='Each line is a SQL predicate for a subset to analyze.'
    ).split('\n')

st.markdown('---')

if st.button('Find Pattern!', help='Start mining for the described pattern.'):
    st.warning('Mining in progress... This may take a moment.')
    result_cols = ['Predicate', 'Facts', 'Quality']
    results = []
    for cmp_pred in cmp_preds:
        st.write(f'Analyzing data satisfying predicate "{cmp_pred}" ...')
        dims_col_text = [d.split(':') for d in dims_info]
        aggs_col_text = [a.split(':') for a in aggs_info]
        t = {
            'table':table, 
            'dim_cols':[d[0] for d in dims_col_text],
            'agg_cols':[a[0] for a in aggs_col_text],
            'cmp_preds':[cmp_pred],
            'nr_facts':nr_facts, 'nr_preds':nr_preds, 
            'degree':5, 'max_steps':nr_iterations,
            'preamble':preamble, 
            'dims_tmp':[d[1] for d in dims_col_text],
            'aggs_txt':[a[1] for a in aggs_col_text]
        }
        with psycopg2.connect(
            database=db_name, user=db_user, 
            cursor_factory=psycopg2.extras.RealDictCursor) as connection:
            connection.autocommit = True
            start_s = time.time()
            all_preds = nminer.sql.pred.all_preds(
                connection, t['table'], 
                t['dim_cols'], cmp_pred)
            sum_eval = nminer.text.sum.SumEvaluator(
                1, 'facebook/bart-large-mnli', label)
            env = nminer.algs.rl.PickingEnv(
                connection, **t, all_preds=all_preds,
                c_type='proactive', cluster=True,
                sum_eval=sum_eval)
            model = stable_baselines3.A2C(
                'MlpPolicy', env, verbose=True, 
                gamma=1.0, normalize_advantage=True)
            model.learn(total_timesteps=nr_iterations)
            rated_sums = env.s_eval.text_to_reward
            actual_sums = [i for i in rated_sums.items() if i[0] is not None]
            sorted_sums = sorted(actual_sums, key=lambda s: s[1])
            if sorted_sums:
                b_sum = sorted_sums[-1]
            else:
                b_sum = ('(No valid summary generated)', -10)
            results.append([cmp_pred, b_sum[0], b_sum[1]])
    result_df = pd.DataFrame(results, columns=result_cols)
    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
    st.dataframe(result_df)
    st.success('All data subsets generated!')

st.markdown('---')