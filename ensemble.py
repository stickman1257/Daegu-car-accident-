import numpy as np
import pandas as pd



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




