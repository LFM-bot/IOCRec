import pickle
import logging
import datetime
import torch
import os
import time as t
import numpy as np


class Recorder:
    def __init__(self, config):
        self.epoch = 0
        self.model_name = config.model
        self.dataset = config.dataset
        self.run_mark = config.mark
        self.log_path = config.log_save
        # metric
        self.metrics = config.metric
        self.k_list = config.k
        # records
        self.batch_loss_rec = 0.
        self.metric_records = {}
        self.time_record = {'train': 0., 'eval': 0.}
        self.decimal_round = 4
        self.mark = config.mark
        self.model_saved = config.model_saved

        # early stop
        self.early_stop = False
        self.core_metric = config.valid_metric
        self.patience = int(config.patience)
        self.best_metric_rec = {'epoch': 0, 'score': 0.}
        self.step_2_stop = self.patience
        # log report
        self.block_size = 6
        self.half_underline = self.block_size * len(self.metrics) * len(self.k_list)

        self._recoder_init(config)

    def reset(self):
        self.epoch = 0

    def _recoder_init(self, config):
        self._init_log()
        self._init_record()
        self._model_saving_init(config)

    def _model_saving_init(self, config):
        # check saving path
        if not os.path.exists(config.save):
            os.mkdir(config.save)
        # init model saving path
        curr_time = datetime.datetime.now()
        timestamp = datetime.datetime.strftime(curr_time, '%Y-%m-%d_%H-%M-%S')
        if self.model_saved is None:
            self.model_saved = config.save + f'\\{config.model}-{self.dataset}-{self.mark}-{timestamp}.pth'
        logging.info(f'model save at: {self.model_saved}')

    def _init_log(self):
        save_path = os.path.join(self.log_path, self.dataset)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        times = 1
        log_model_name = self.model_name + f'-{self.run_mark}' if len(self.run_mark) > 0 else self.model_name
        log_file = os.path.join(save_path, '%s_%d.log' % (log_model_name, times))
        for i in range(100):
            if not os.path.isfile(log_file):
                break
            log_file = os.path.join(save_path, '%s_%d.log' % (log_model_name, times + i + 1))

        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        logging.info('log save at : {}'.format(log_file))

    def _init_record(self):
        for metric in self.metrics:
            for k in self.k_list:
                self.metric_records[f'{metric}@{k}'] = []
        assert self.core_metric in self.metric_records.keys(), f'Invalid valid_metric: [{self.core_metric}], ' \
                                                               f'choose from: {self.metric_records.keys()} !'

    def save_model(self, model):
        # save entire model
        # torch.save(model, self.model_saving)

        # only save model parameters
        torch.save(model.state_dict(), self.model_saved)

    def load_best_model(self, model):
        # load entire model
        # return torch.load(self.model_saving)

        # load parameters
        model.load_state_dict(torch.load(self.model_saved))

    def epoch_restart(self):
        self.batch_loss_rec = 0.
        self.epoch += 1

    def save_batch_loss(self, batch_loss):
        self.batch_loss_rec += batch_loss

    def tik_start(self):
        self._clock = t.time()

    def tik_end(self, mode='train'):
        end_clock = t.time()
        self.time_record[mode] = end_clock - self._clock

    def _save_best_result(self, metric_res, model):
        """
        :param metric_res: dict
        """
        for metric, score in metric_res.items():
            self.metric_records.get(metric).append(score)
        # early stop
        core_metric_res = metric_res.get(self.core_metric)
        self.early_stop_check(core_metric_res, model)

    def early_stop_check(self, core_metric_res, model):
        if core_metric_res > self.best_metric_rec.get('score'):
            self.best_metric_rec['score'] = core_metric_res
            self.best_metric_rec['epoch'] = self.epoch
            self.step_2_stop = self.patience
            # find a better model -> save
            self.save_model(model)
        else:
            self.step_2_stop -= 1
            logging.info(f'EarlyStopping Counter: {self.patience - self.step_2_stop} out of {self.patience}')
        if self.step_2_stop == 0:
            self.early_stop = True

    def train_log_verbose(self, num_batch):
        training_loss = self.batch_loss_rec / num_batch
        logging.info('-' * self.half_underline + f'----Epoch {self.epoch}----' + '-' * self.half_underline)
        output_str = " Training Time :[%.1f s]\tTraining Loss = %.4f" % (self.time_record['train'], training_loss)
        logging.info(output_str)

    def log_verbose_and_save(self, metric_score, eval_loss, model):
        res_str = ''
        for metric, score in metric_score.items():
            score = round(score, self.decimal_round)
            res_str += f'{metric}:{score:1.4f}\t'

        eval_time = round(self.time_record['eval'], 1)
        if eval_loss <= 0:
            eval_loss = '**'
        else:
            eval_loss = round(eval_loss, 4)
        logging.info(f"Evaluation Time:[{eval_time} s]\t  Eval Loss   = {eval_loss}")
        logging.info(res_str)

        # save results and model
        self._save_best_result(metric_score, model)

    def report_best_res(self):
        best_epoch = self.best_metric_rec['epoch']
        logging.info('-' * self.half_underline + 'Best Evaluation' + '-' * self.half_underline)
        logging.info(f"Best Result at Epoch: {best_epoch}\t Early Stop at Patience: {self.patience}")
        # load best results
        best_metrics_res = {}
        for metric, metric_res_list in self.metric_records.items():
            best_metrics_res[metric] = metric_res_list[best_epoch - 1]
        res_str = ''
        for metric, score in best_metrics_res.items():
            score = round(score, self.decimal_round)
            res_str += f'{metric}:{score:1.4f}\t'
        logging.info(res_str)

    def report_test_result(self, test_metric_res):
        res_str = ''
        for metric, score in test_metric_res.items():
            score = round(score, self.decimal_round)
            res_str += f'{metric}:{score:1.4f}\t'
        logging.info('-' * self.half_underline + f'-----Test Results------' + '-' * self.half_underline)
        logging.info(res_str)