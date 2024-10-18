import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import levene, mannwhitneyu, f_oneway

clients = pd.read_csv('/datasets/telecom_clients_us.csv')
telecom = pd.read_csv('/datasets/telecom_dataset_us.csv')

#Clients table
clients.head()
clients.info()
clients['date_start'] = pd.to_datetime(clients['date_start'], format = '%Y-%m-%d')
clients[clients.duplicated()]

#telecom table
telecom.head()
telecom.info()
telecom['date'] = pd.to_datetime(telecom['date'], errors='coerce')
telecom['date'] = telecom['date'].dt.tz_localize(None)
telecom[telecom.duplicated()]
telecom = telecom.drop_duplicates()
telecom[telecom['operator_id']==999]
telecom['operator_id'].fillna(999, inplace=True)

#ANALYSIS
#Missed calls
missed_calls = telecom[telecom['direction'] == 'in']
missed_calls = missed_calls.pivot_table(index='operator_id', columns='is_missed_call', aggfunc='size').reset_index()
missed_calls.head()
missed_calls.fillna(0, inplace=True)
missed_calls['miss_percent'] = ((missed_calls[True] / (missed_calls[False] + missed_calls[True])) * 100).round(3)
missed_calls['miss_percent'].describe()

#Histogram of missed calls
sns.set_theme(style='darkgrid')
plt.figure()
sns.histplot(data = missed_calls[missed_calls['operator_id']!=999], x = 'miss_percent', bins=10)
plt.xlabel('Missed pertentage')
plt.title('Histogram of missed calls')
plt.show()

def rating_column(df, base_col, new_col, good, attention):
    ratings = []
    for value in df[base_col]:
        if value < good:
            ratings.append('good')
        elif good <= value <= attention:
            ratings.append('attention')
        else:
            ratings.append('inefficient')
    df[new_col] = ratings
    return df

rating_column(missed_calls,'miss_percent','miss_rating', 10, 20)

#Time on hold
hold_times = telecom.groupby('operator_id', as_index = False)['calls_count', 'call_duration', 'total_call_duration'].sum()
hold_times.head()
hold_times['hold_time'] = ((hold_times['total_call_duration'] - hold_times['call_duration']) / hold_times['calls_count']).round(3)
hold_times.head()
hold_times['hold_time'].describe()

#Histogram of hold time
plt.figure()
sns.histplot(data = hold_times, x = 'hold_time',bins=10)
plt.xlabel('Time on Hold (seconds)')
plt.title('Distribution of Hold Time')
plt.show()

rating_column(hold_times, 'hold_time', 'hold_rating', 16, 30)

#Total daily calls
total_calls = telecom.groupby('operator_id').agg(min_date=('date', 'min'),
                                                     max_date=('date', 'max'),
                                                     total_calls=('calls_count', 'sum')).reset_index()
total_calls['active_days'] = ((total_calls['max_date'] - total_calls['min_date']).dt.days) + 1
total_calls['daily_calls'] = (total_calls['total_calls'] / total_calls['active_days']).round(2)
total_calls.head()

#For this section, we removed the fictitious operator since it includes the sum of several unknowns and produces an outlier, skewing the data.
total_calls = total_calls[total_calls['operator_id']!=999]
total_calls['daily_calls'].describe()

#Filter extreme values
filter_calls = total_calls[total_calls['daily_calls'] < total_calls['daily_calls'].quantile(0.95)]
filter_calls['daily_calls'].describe()

#Histogram of daily calls by operator
plt.figure()
sns.histplot(data = filter_calls, x = 'daily_calls')
plt.xlabel('Daily Calls')
plt.ylabel('Operators count')
plt.title('Count of Daily Calls by Operator')
plt.show()

rating_column(filter_calls, 'daily_calls', 'n_calls_rating', 7, 10)
#We got opposite results so have to change it.
filter_calls['n_calls_rating'] = filter_calls['n_calls_rating'].replace({'inefficient': 'good', 'good': 'inefficient'})

#Call duration
call_duration = hold_times.iloc[:, :3]
call_duration['mean_duration'] = (call_duration['call_duration'] / call_duration['calls_count']).round(0)
call_duration['mean_duration'].describe()

#Distribution of data
sns.boxplot(data=call_duration, x='mean_duration')
call_duration['mean_duration'].hist(range=(0,400))

#Define ranges and adapt the function
q_5 = call_duration['mean_duration'].quantile(0.05)
q_15 = call_duration['mean_duration'].quantile(0.15)
q_85 = call_duration['mean_duration'].quantile(0.85)
q_95 = call_duration['mean_duration'].quantile(0.95)

ratings = []

for x in call_duration['mean_duration']:
    if x <= q_5 or x >= q_95:
        ratings.append('inefficient')
    elif (q_5 < x <= q_15) or (q_85 <= x < q_95):
        ratings.append('attention')
    else:
        ratings.append('good')

call_duration['duration_rating'] = ratings
call_duration.head()

#Mean call duration
plt.figure()
sns.histplot(data= call_duration, x = 'mean_duration', binrange=(0,400))
plt.xlabel('Call duration (seconds)')
plt.ylabel('Operators count')
plt.title('Mean Call Duration')
plt.show()

#Combine and add scores
op_rating = missed_calls.iloc[:, [0, -1]].merge(hold_times.iloc[:, [0, -1]], on='operator_id', how='outer')
op_rating = op_rating.merge(filter_calls.iloc[:, [0, -1]], on = 'operator_id', how = 'outer')
op_rating = op_rating.merge(call_duration.iloc[:, [0, -1]], on = 'operator_id', how = 'outer')
op_rating.info()

#Add the score of 'filtered' operators
op_rating = op_rating.fillna('good')

"""
Cambio los resultados por valores numéricos para sumar la puntuación final y en base a eso se asigna cada operador a una categoría descrita a continuación:
Puntaje de 0 a 20 en la que 
- 16 o más puntos (Good): No es ineficiente en ningún area, como mucho requiere atención en alguna.
- Entre 11 y 15 (Attention): En general no es particularmente malo pero puede mejorar.
- Hasta 10 (Inefficient): Operadores con muy mal desempeño, requieren atención inmediata. 
"""

op_score = op_rating
score_map = {'good': 5, 'attention': 2, 'inefficient': 0}
op_score = op_score.replace(score_map)
op_score['score'] = op_score[['miss_rating', 'hold_rating', 'n_calls_rating', 'duration_rating']].sum(axis=1)

final_rating = []

for x in op_score['score']:
    if x >= 16:
        final_rating.append('good')
    elif 11 <= x < 16:
        final_rating.append('attention')
    else:
        final_rating.append('inefficient')

op_score['final_rating'] = final_rating
op_score.head()

op_score['final_rating'].value_counts()

#Score distribution
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(data = op_score, x = 'score', ax = axs[0])
axs[0].set_title('Numeric Score Distribution')
axs[0].set_xlabel('Score')
axs[0].set_ylabel('Operators Count')

sns.histplot(data = op_score, x = 'final_rating', ax = axs[1])
axs[1].set_title('Rating Distribution')
axs[1].set_xlabel('Final Rating')
axs[1].set_ylabel('Operators Count')

plt.tight_layout()
plt.show()

#Hypothesis
# 1: Good operators make more external calls
alpha = 0.05

rating_tlc = telecom.merge(op_score[['operator_id','final_rating']], on ='operator_id', how='left')

out_calls = rating_tlc[rating_tlc['direction'] == 'out']

calls_summary = out_calls.groupby('operator_id')['calls_count'].sum().reset_index()
calls_summary = calls_summary.merge(op_score[['operator_id','final_rating']], on='operator_id', how='left')

# Separar por clasificación
good_operators = calls_summary[calls_summary['final_rating'] == 'good']
inefficient_operators = calls_summary[calls_summary['final_rating'] == 'inefficient']

# Levene para evaluar varianzas 
levene_stat, levene_p_value = levene(good_operators['calls_count'], inefficient_operators['calls_count'])
print(f'Estadístico de Levene: {levene_stat}')
print(f'Valor p de Levene: {levene_p_value}')

if levene_p_value < alpha:
    print("Las varianzas son significativamente diferentes.")
else:
    print("No hay evidencia suficiente para afirmar que las varianzas son diferentes.")
    
    """
Estadístico de Levene: 35.975455509555154
Valor p de Levene: 4.1599001756503035e-09
Las varianzas son significativamente diferentes.
    """
#Mann-Whitney U test
u_stat, mannwhitney_p_value = mannwhitneyu(good_operators['calls_count'], inefficient_operators['calls_count'], alternative='greater')

print(f'Estadístico U: {u_stat}')
print(f'Valor p de Mann-Whitney: {mannwhitney_p_value}')

if mannwhitney_p_value < alpha:
    print("Rechazamos la H0, los operadores buenos realizan significativamente más llamadas.")
else:
    print("No hay suficiente evidencia para afirmar que hay diferencia en las llamadas de ambos grupos.")

"""
Estadístico U: 42369.0
Valor p de Mann-Whitney: 1.3120492578479738e-56
Rechazamos la H0, los operadores buenos realizan significativamente más llamadas.
"""

#2: Good operators favor a specific tariff plan
plan_count = clients['tariff_plan'].value_counts()
plan_dict = dict(plan_count)

plan_tlc = rating_tlc.merge(clients[['user_id','tariff_plan']], on='user_id', how='left')
good_interactions = plan_tlc[plan_tlc['final_rating'] == 'good']

# Agrupar por plan y contar las interacciones
plan_a = good_interactions[good_interactions['tariff_plan'] == 'A']['calls_count']
plan_b = good_interactions[good_interactions['tariff_plan'] == 'B']['calls_count']
plan_c = good_interactions[good_interactions['tariff_plan'] == 'C']['calls_count']

# Aplicar ANOVA
f_stat, p_value = f_oneway(plan_a, plan_b, plan_c)
print(f'Estadístico F: {f_stat}')
print(f'Valor p: {p_value}')

if p_value < alpha:
    print("Se rechaza la hipótesis nula. Hay diferencias significativas entre los planes.")
else:
    print("No hay suficiente evidencia para rechazar H0")

"""
    Estadístico F: 149.7773672106206
Valor p: 2.721535337357596e-65
Se rechaza la hipótesis nula. Hay diferencias significativas entre los planes.
"""

#Find the preferred plan
plan_count = clients['tariff_plan'].value_counts()
plan_dict = dict(plan_count)

interaction = good_interactions.groupby('tariff_plan')['user_id'].count().reset_index()
interaction

#Calls by plan
interaction['plan_count'] = interaction['tariff_plan'].map(plan_dict)
interaction['prop_interactions'] = interaction['user_id'] / interaction['plan_count']

plt.figure()
sns.barplot(data = interaction, x = 'tariff_plan', y = 'prop_interactions')
plt.xlabel('Plan')
plt.ylabel('Interactions')
plt.title('Llamadas de operadores eficaces por plan')


""" 
Conclusiones y recomendaciones

Se ha logrado el objetivo de categorizar el rendimiento de operadores para identificar a los deficientes así como los que están en riesgo y se establecieron umbrales de lo que es un operador eficaz para aspirar a esas métricas.

### Recomendaciones:
**Mejorar la capacitación:**
Se deben implementar programas de capacitación continua para los operadores, con refuerzo en los "ineficientes"para ayudar a mejorar su rendimiento y aumentar la satisfacción del cliente.

**Evaluar el servicio al cliente del plan A:**
Dado que los usuarios del plan A tienen mayor interacción con operadores eficientes, se recomienda evaluar las razones detrás de esto. Podría ser útil recopilar retroalimentación de los usuarios de este plan para identificar áreas de mejora.

**Monitoreo continuo y mejora:**
Implementar un sistema de monitoreo continuo para evaluar el rendimiento de los operadores. Esto incluiría métricas de eficiencia, tiempos de respuesta y satisfacción del cliente. Con estos datos, se pueden hacer ajustes en tiempo real para optimizar la atención al cliente.

**Incentivos por buen rendimiento:**
Considerar la implementación de promociones o incentivos específicos para los operadores con buena evaluación para fomentar interacciones más efectivas y la adopción de mejores prácticas.


El análisis realizado sugiere que existen áreas significativas de mejora en la interacción entre los operadores y los usuarios. Implementar las recomendaciones proporcionadas no solo puede ayudar a identificar operadores ineficaces, sino también a mejorar la experiencia del cliente en general, lo que es fundamental para el éxito a largo plazo en la industria de telecomunicaciones.
"""
# Results: https://drive.google.com/file/d/1w2WHftugqI0TsIefJUiucbQGRyMMpn5e/view?usp=sharing