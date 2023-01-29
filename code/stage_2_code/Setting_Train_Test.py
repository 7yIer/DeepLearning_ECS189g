'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np


class Setting_Train_Test(setting):
    fold = 3

    def load_run_save_evaluate(self):
        # load dataset
        loaded_data, loaded_data_test = self.dataset.load()

        self.method.data = {'train': {'X': loaded_data['X'], 'y': loaded_data['y']}, 'test': {'X': loaded_data_test['X'], 'y': loaded_data_test['y']}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        score = self.evaluate.evaluate()

        return score, 1

