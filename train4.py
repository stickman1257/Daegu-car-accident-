import os
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)



light_df = pd.read_csv('../data/open/external_open/대구 보안등 정보.csv', encoding='cp949')[['설치개수', '소재지지번주소']]

location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

light_df[['도시', '구', '동', '번지']] = light_df['소재지지번주소'].str.extract(location_pattern)
light_df = light_df.drop(columns=['소재지지번주소', '번지'])

light_df = light_df.groupby(['도시', '구', '동']).mean().reset_index()
light_df.reset_index(inplace=True, drop=True)

child_area_df = pd.read_csv('../data/open/external_open/대구 어린이 보호 구역 정보.csv', encoding='cp949').drop_duplicates()[['소재지지번주소']]
child_area_df['cnt'] = 1

location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

child_area_df[['도시', '구', '동', '번지']] = child_area_df['소재지지번주소'].str.extract(location_pattern)
child_area_df = child_area_df.drop(columns=['소재지지번주소', '번지'])

child_area_df = child_area_df.groupby(['도시', '구', '동']).mean().reset_index()
child_area_df.reset_index(inplace=True, drop=True)

parking_df = pd.read_csv('../data/open/external_open/대구 주차장 정보.csv', encoding='cp949')[['소재지지번주소', '급지구분']]
parking_df = pd.get_dummies(parking_df, columns=['급지구분'])

location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

parking_df[['도시', '구', '동', '번지']] = parking_df['소재지지번주소'].str.extract(location_pattern)
parking_df = parking_df.drop(columns=['소재지지번주소', '번지'])

parking_df = parking_df.groupby(['도시', '구', '동']).mean().reset_index()
parking_df.reset_index(inplace=True, drop=True)

train_org = pd.read_csv('../data/open/train.csv') 
test_org = pd.read_csv('../data/open/test.csv')
countrywide_org = pd.read_csv('../data/open/external_open/countrywide_accident.csv')

train_df = train_org.copy()
test_df = test_org.copy()
countrywide_df = countrywide_org.copy()

time_pattern = r'(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2})' 

train_df[['연', '월', '일', '시간']] = train_org['사고일시'].str.extract(time_pattern)
train_df[['연', '월', '일', '시간']] = train_df[['연', '월', '일', '시간']].apply(pd.to_numeric) # 추출된 문자열을 수치화해줍니다 
train_df = train_df.drop(columns=['사고일시']) # 정보 추출이 완료된 '사고일시' 컬럼은 제거합니다 

countrywide_df[['연', '월', '일', '시간']] = countrywide_df['사고일시'].str.extract(time_pattern)  #org와 정제한 df는 value의 순서와 개수가 다르다.
countrywide_df[['연', '월', '일', '시간']] = countrywide_df[['연', '월', '일', '시간']].apply(pd.to_numeric)
countrywide_df = countrywide_df.drop(columns=['사고일시']) 
# 해당 과정을 test_x에 대해서도 반복해줍니다 
test_df[['연', '월', '일', '시간']] = test_org['사고일시'].str.extract(time_pattern)
test_df[['연', '월', '일', '시간']] = test_df[['연', '월', '일', '시간']].apply(pd.to_numeric)
test_df = test_df.drop(columns=['사고일시'])

location_pattern = r'(\S+) (\S+) (\S+)'

train_df[['도시', '구', '동']] = train_org['시군구'].str.extract(location_pattern)
train_df = train_df.drop(columns=['시군구'])
test_df[['도시', '구', '동']] = test_org['시군구'].str.extract(location_pattern)
test_df = test_df.drop(columns=['시군구'])


countrywide_df[['도시', '구', '동']] = countrywide_org['시군구'].str.extract(location_pattern)
nan_country_df = countrywide_df[pd.isna(countrywide_df['도시'])] #세종시 XX동 정제 작업을 위해 진행.
countrywide_df = countrywide_df.dropna(subset=['도시'])

location_pattern = r'(\S+) (\S+)'
nan_country_df[['도시', '동']] = countrywide_org['시군구'].str.extract(location_pattern)
countrywide_df = pd.concat([nan_country_df, countrywide_df], ignore_index=True)  #정제 작업 후 countrywide_df에 하나로 합침.
countrywide_df = countrywide_df.drop(columns=['시군구'])


road_pattern = r'(.+) - (.+)'

train_df[['도로형태1', '도로형태2']] = train_org['도로형태'].str.extract(road_pattern)
train_df = train_df.drop(columns=['도로형태'])

countrywide_df[['도로형태1', '도로형태2']] = countrywide_df['도로형태'].str.extract(road_pattern)
countrywide_df = countrywide_df.drop(columns=['도로형태'])

test_df[['도로형태1', '도로형태2']] = test_org['도로형태'].str.extract(road_pattern)
test_df = test_df.drop(columns=['도로형태'])

# train_df와 test_df에, light_df와 child_area_df, parking_df를 merge하세요.
train_df = pd.merge(train_df, light_df, how='left', on=['도시', '구', '동'])
train_df = pd.merge(train_df, child_area_df, how='left', on=['도시', '구', '동'])
train_df = pd.merge(train_df, parking_df, how='left', on=['도시', '구', '동'])

test_df = pd.merge(test_df, light_df, how='left', on=['도시', '구', '동'])
test_df = pd.merge(test_df, child_area_df, how='left', on=['도시', '구', '동'])
test_df = pd.merge(test_df, parking_df, how='left', on=['도시', '구', '동'])

train_df = pd.concat([train_df, countrywide_df], ignore_index=False)

test_x = test_df.drop(columns=['ID']).copy()
train_x = train_df[test_x.columns].copy()
train_y = train_df['ECLO'].copy()


from category_encoders.target_encoder import TargetEncoder

categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)



for i in categorical_features:
    le = TargetEncoder(cols=[i])
    train_x[i] = le.fit_transform(train_x[i], train_y)
    test_x[i] = le.transform(test_x[i])


train_x.fillna(0, inplace=True)
test_x.fillna(0, inplace=True)




import tensorflow as tf


from tensorflow.python.client import device_lib


tf.config.list_physical_devices('GPU')

def rmsle(y_true, y_pred):
    y_true = tf.maximum(tf.cast(y_true, tf.float32), 0)
    y_pred = tf.maximum(tf.cast(y_pred, tf.float32), 0)
    
    squared_error = tf.square(tf.math.log1p(y_pred) - tf.math.log1p(y_true))

    return tf.sqrt(tf.reduce_mean(squared_error))

def loss_fn(y_true, y_pred):
    return rmsle(y_true, y_pred)

def metric_fn(y_true, y_pred):
    return rmsle(y_true, y_pred)
callbacks_list = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=2, mode='min',restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.00001),
    tf.keras.callbacks.ModelCheckpoint(filepath='./best_model.h5', monitor='val_loss', save_best_only=True, mode='min'),
    tf.keras.callbacks.TerminateOnNaN()
] 
best_params = {'batch_size': 128, 'l1_reg': 0.0001, 'l2_reg': 0.0001, 'learning_rate': 0.01}
def create_model(l1_reg, l2_reg, learning_rate):
    
    input_layer = tf.keras.Input(shape=(len(train_x.columns), ))
    x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(input_layer)
    

    x = tf.keras.layers.Dense(32, kernel_regularizer= tf.keras.regularizers.l1(l1_reg))(x)      
    x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(64, kernel_regularizer= tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=0.00001)(x)
    x = tf.keras.layers.Activation('relu')(x)
    output_layer = tf.keras.layers.Dense(1)(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=loss_fn,
                  metrics=[metric_fn],
                  
                  )
    
    return model
from sklearn.model_selection import StratifiedKFold


test_preds = np.zeros((len(test_x), 1))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in skf.split(train_x.values, train_y.values):
    train_x_fold, val_x_fold = train_x.iloc[train_index], train_x.iloc[val_index]
    train_y_fold, val_y_fold = train_y.iloc[train_index], train_y.iloc[val_index]
    
    model = create_model(best_params['l1_reg'], best_params['l2_reg'], best_params['learning_rate'])
    history = model.fit(train_x_fold.astype('float32'), train_y_fold.astype('float32'),
                        epochs=150,
                        callbacks=callbacks_list,
                        batch_size=best_params['batch_size'],
                        validation_data=(val_x_fold.astype('float32'), val_y_fold.astype('float32')))
    test_preds += model.predict(test_x.values) / skf.n_splits


import matplotlib.pyplot as plt

plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

sample_submission = pd.read_csv('../data/open/sample_submission.csv')

sample_submission["ECLO"] = model.predict(test_x.astype('float32'))

sample_submission.to_csv("./sub/tf_utimate_submission2.csv", index=False)

csv_files = [
              './ensemble/submit.csv',
              './ensemble/tf_submission7.csv',
              './ensemble/tf_utimate_submission1.csv'

            ]

data_list = [pd.read_csv(file)['ECLO'] for file in csv_files]

common_columns = set.intersection(*[set(df.columns) if isinstance(df, pd.DataFrame) else set([df.name]) for df in data_list])


average_values = sum([df[common_columns].mean(axis=1) if isinstance(df, pd.DataFrame) else df for df in data_list]) / len(data_list)
sample_submission = pd.read_csv('../data/open/sample_submission.csv')

sample_submission["ECLO"] = average_values

sample_submission.to_csv("./ensemble/ensemble_submission03.csv", index=False)


