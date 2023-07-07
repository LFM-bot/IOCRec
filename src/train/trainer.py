import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import dataset
from src.dataset.dataset import load_specified_dataset
from src.dataset.data_processor import DataProcessor
from src.evaluation.estimator import Estimator
from src.utils.recorder import Recorder
import src.model as model
from src.train.config import experiment_hyper_load, config_override
from src.utils.utils import set_seed, batch_to_device, KMeans


# torch.autograd.set_detect_anomaly(True)

def load_trainer(config):
    if config.model in ['ICLRec']:
        return ICLTrainer(config)
    return Trainer(config)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.model_name = config.model
        self._config_override(self.model_name, config)
        # pretraining
        self.pretraining_model = None
        self.do_pretraining = self.config.do_pretraining
        self.pretraining_task = self.config.pretraining_task
        self.pretraining_epoch = self.config.pretraining_epoch
        self.pretraining_batch = self.config.pretraining_batch
        self.pretraining_lr = self.config.pretraining_lr
        self.pretraining_l2 = self.config.pretraining_l2

        # training
        self.training_model = None
        self.num_worker = self.config.num_worker
        self.train_batch = self.config.train_batch
        self.eval_batch = self.config.eval_batch
        self.lr = self.config.learning_rate
        self.l2 = self.config.l2
        self.epoch_num = self.config.epoch_num
        self.dev = torch.device(self.config.device)
        self.split_type = self._set_split_mode(self.config.split_type)
        self.do_test = self.split_type == 'valid_and_test'

        # components
        self.data_processor = DataProcessor(self.config)
        self.estimator = Estimator(self.config)
        self.recorder = Recorder(self.config)

        # set random seed
        set_seed(self.config.seed)

        # preparing data
        data_dict, additional_data_dict = self.data_processor.prepare_data()
        self.data_dict = data_dict  # store standard train/eval/test data
        self.additional_data_dict = additional_data_dict  # extra data (model specified)

        #  check_duplication(self.data_dict['train'][0])

        self.estimator.load_item_popularity(self.data_processor.popularity)
        self._set_num_items()

    def start_training(self):
        if self.do_pretraining:
            self.pretrain()
        self.train()

    def pretrain(self):
        if self.pretraining_task in ['MISP', 'MIM', 'PID']:
            pretrain_dataset = getattr(dataset, f'{self.pretraining_task}PretrainDataset')
            pretrain_dataset = pretrain_dataset(self.config, self.data_dict['train'],
                                                self.additional_data_dict)
        else:
            raise NotImplementedError(f'No such pretraining task: {self.pretraining_task}, '
                                      f'choosing from [MIP, MIM, PID]')
        train_loader = DataLoader(pretrain_dataset, batch_size=self.train_batch, collate_fn=pretrain_dataset.collate_fn,
                                  shuffle=True, num_workers=0, drop_last=False)

        pretrain_model = self._load_model()

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, pretrain_model.parameters()),
                               self.pretraining_lr, weight_decay=self.pretraining_l2)

        self.experiment_setting_verbose(pretrain_model, training=False)

        logging.info('Start pretraining...')
        for epoch in range(self.pretraining_epoch):
            pretrain_model.train()
            self.recorder.epoch_restart()
            self.recorder.tik_start()
            train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
            train_iter.set_description(f'pretraining  epoch: {epoch}')
            for i, batch_dict in train_iter:
                batch_to_device(batch_dict, self.dev)
                loss = getattr(pretrain_model, f'{self.pretraining_task}_pretrain_forward')(batch_dict)
                opt.zero_grad()
                loss.backward()
                opt.step()

                self.recorder.save_batch_loss(loss.item())
            self.recorder.tik_end()
            self.recorder.train_log_verbose(len(train_loader))

        self.pretraining_model = pretrain_model
        logging.info('Pre-training is over, prepare for training...')

    def train(self):
        SpecifiedDataSet = load_specified_dataset(self.model_name, self.config)
        train_dataset = SpecifiedDataSet(self.config, self.data_dict['train'],
                                         self.additional_data_dict)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch, collate_fn=train_dataset.collate_fn,
                                  shuffle=True, num_workers=self.num_worker, drop_last=False)

        eval_dataset = SpecifiedDataSet(self.config, self.data_dict['eval'],
                                        self.additional_data_dict, train=False)
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch, collate_fn=eval_dataset.collate_fn,
                                 shuffle=False, num_workers=self.num_worker, drop_last=False)

        self.training_model = self._load_model()

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.training_model.parameters()), self.lr,
                               weight_decay=self.l2)
        self.recorder.reset()
        self.experiment_setting_verbose(self.training_model)

        logging.info('Start training...')
        for epoch in range(self.epoch_num):
            self.training_model.train()
            self.recorder.epoch_restart()
            self.recorder.tik_start()
            train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
            train_iter.set_description('training  ')
            for step, batch_dict in train_iter:
                # training forward
                batch_dict['epoch'] = epoch
                batch_dict['step'] = step
                batch_to_device(batch_dict, self.dev)
                loss = self.training_model.train_forward(batch_dict)
                if torch.is_tensor(loss) and loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    self.recorder.save_batch_loss(loss.item())
            self.recorder.tik_end()
            self.recorder.train_log_verbose(len(train_loader))

            # evaluation
            self.recorder.tik_start()
            eval_metric_result, eval_loss = self.estimator.evaluate(eval_loader, self.training_model)
            self.recorder.tik_end(mode='eval')
            self.recorder.log_verbose_and_save(eval_metric_result, eval_loss, self.training_model)

            if self.recorder.early_stop:
                break

        self.recorder.report_best_res()
        # test model
        if self.do_test:
            test_metric_res = self.test_model(self.data_dict['test'])
            self.recorder.report_test_result(test_metric_res)

    def _set_split_mode(self, split_mode):
        assert split_mode in ['valid_and_test', 'valid_only'], f'Invalid split mode: {split_mode} !'
        return split_mode

    def _load_model(self):
        if self.do_pretraining and self.pretraining_model is not None:  # return pretraining model
            return self.pretraining_model

        # return new model
        if self.config.model_type.upper() == 'SEQUENTIAL':
            return self._load_sequential_model()
        elif self.config.model_type.upper() in ['GRAPH', 'KNOWLEDGE']:
            return self._load_model_with_additional_data()
        else:
            raise KeyError(f'Invalid model_type:{self.config.model_type}. Choose from [sequential, knowledge, graph]')

    def _load_sequential_model(self):
        Model = getattr(model, self.model_name)
        specified_seq_model = Model(self.config, self.additional_data_dict).to(self.dev)
        return specified_seq_model

    def _load_model_with_additional_data(self):
        Model = getattr(model, self.model_name)
        specified_model = Model(self.config, self.additional_data_dict).to(self.dev)
        return specified_model

    def _config_override(self, model_name, cmd_config):
        self.model_config = getattr(model, f'{model_name}_config')()
        self.config = config_override(self.model_config, cmd_config)
        # capitalize
        self.config.model_type = self.config.model_type.upper()
        self.config.graph_type = [g_type.upper() for g_type in self.config.graph_type]

    def _set_num_items(self):
        self.config.num_items = self.data_processor.num_items

    def experiment_setting_verbose(self, model, training=True):
        if self.do_pretraining and training:
            return
        # model config
        logging.info('[1] Model Hyper-Parameter '.ljust(47, '-'))
        model_param_set = self.model_config.keys()
        for arg in vars(self.config):
            if arg in model_param_set:
                logging.info(f'{arg}: {getattr(self.config, arg)}')
        # experiment config
        logging.info('[2] Experiment Hyper-Parameter '.ljust(47, '-'))
        # verbose_order = ['Data', 'Training', 'Evaluation', 'Save']
        hyper_types, exp_setting = experiment_hyper_load(self.config)
        for i, hyper_type in enumerate(hyper_types):
            hyper_start_log = (f'[2-{i + 1}] ' + hyper_type.lower() + ' hyper-parameter ').ljust(47, '-')
            logging.info(hyper_start_log)
            for hyper, value in exp_setting[hyper_type].items():
                logging.info(f'{hyper}: {value}')
        # data statistic
        self.data_processor.data_log_verbose(3)
        # model architecture
        self.report_model_info(model)

    def report_model_info(self, model):
        # model architecture
        logging.info('[1] Model Architecture '.ljust(47, '-'))
        logging.info(f'total parameters: {model.calc_total_params()}')
        logging.info(model)

    def test_model(self, test_data_pair=None):
        SpecifiedDataSet = load_specified_dataset(self.model_name, self.config)
        test_dataset = SpecifiedDataSet(self.config, test_data_pair,
                                        self.additional_data_dict, train=False)
        test_loader = DataLoader(test_dataset, batch_size=self.eval_batch, num_workers=self.num_worker,
                                 collate_fn=test_dataset.collate_fn, drop_last=False, shuffle=False)
        # load the best model
        self.recorder.load_best_model(self.training_model)
        self.training_model.eval()
        test_metric_result = self.estimator.test(test_loader, self.training_model)

        return test_metric_result

    def start_test(self):
        self.training_model = self._load_model()
        self.experiment_setting_verbose(self.training_model)
        test_metric_res = self.test_model(self.data_dict['test'])
        self.recorder.report_test_result(test_metric_res)


class ICLTrainer(Trainer):
    def __init__(self, config):
        super(ICLTrainer, self).__init__(config)
        self.num_intent_cluster = config.num_intent_cluster
        self.seq_representation_type = config.seq_representation_type
        # initialize Kmeans
        if self.seq_representation_type == "mean":
            cluster = KMeans(
                num_cluster=self.num_intent_cluster,
                seed=self.config.seed,
                hidden_size=self.config.embed_size,
                device=self.config.device,
            )
        else:
            cluster = KMeans(
                num_cluster=self.num_intent_cluster,
                seed=self.config.seed,
                hidden_size=self.config.embed_size * self.config.max_len,
                device=self.config.device,
            )
        self.cluster = cluster

    def train(self):
        SpecifiedDataSet = load_specified_dataset(self.model_name, self.config)
        intent_cluster_dataset = SpecifiedDataSet(self.config, self.data_dict['raw_train'],
                                                  self.additional_data_dict)
        intent_cluster_loader = DataLoader(intent_cluster_dataset, batch_size=self.train_batch,
                                           collate_fn=intent_cluster_dataset.collate_fn,
                                           shuffle=True, num_workers=self.num_worker, drop_last=False)

        train_dataset = SpecifiedDataSet(self.config, self.data_dict['train'],
                                         self.additional_data_dict)
        train_loader = DataLoader(train_dataset, batch_size=self.train_batch, collate_fn=train_dataset.collate_fn,
                                  shuffle=True, num_workers=self.num_worker, drop_last=False)

        eval_dataset = SpecifiedDataSet(self.config, self.data_dict['eval'],
                                        self.additional_data_dict, train=False)
        eval_loader = DataLoader(eval_dataset, batch_size=self.eval_batch, collate_fn=eval_dataset.collate_fn,
                                 shuffle=False, num_workers=self.num_worker, drop_last=False)

        self.training_model = self._load_model()

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.training_model.parameters()), self.lr,
                               weight_decay=self.l2)
        self.recorder.reset()
        self.experiment_setting_verbose(self.training_model)

        logging.info('Start training...')
        for epoch in range(self.epoch_num):
            self.training_model.train()
            self.recorder.epoch_restart()
            self.recorder.tik_start()

            # collect cluster data
            intent_cluster_iter = tqdm(enumerate(intent_cluster_loader), total=len(intent_cluster_loader))
            intent_cluster_iter.set_description('prepare clustering  ')

            kmeans_training_data = []  # store all user intent representations in training data
            for step, batch_dict in intent_cluster_iter:
                batch_to_device(batch_dict, self.dev)
                item_seq, seq_len = batch_dict['item_seq'], batch_dict['seq_len']
                sequence_output = self.training_model.seq_encoding(item_seq, seq_len, return_all=True)
                # average sum
                if self.seq_representation_type == "mean":
                    sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                sequence_output = sequence_output.view(sequence_output.shape[0], -1)  # otherwise concat
                sequence_output = sequence_output.detach().cpu().numpy()
                kmeans_training_data.append(sequence_output)
            kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)  # [user_size, dim]

            # train cluster
            self.cluster.train(kmeans_training_data)

            # clean memory
            del kmeans_training_data
            import gc

            gc.collect()

            train_iter = tqdm(enumerate(train_loader), total=len(train_loader))
            train_iter.set_description('training  ')
            for step, batch_dict in train_iter:
                # training forward
                batch_dict['epoch'] = epoch
                batch_dict['step'] = step
                batch_dict['cluster'] = self.cluster
                batch_to_device(batch_dict, self.dev)
                loss = self.training_model.train_forward(batch_dict)
                if torch.is_tensor(loss) and loss.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    self.recorder.save_batch_loss(loss.item())
            self.recorder.tik_end()
            self.recorder.train_log_verbose(len(train_loader))

            # evaluation
            self.recorder.tik_start()
            eval_metric_result, eval_loss = self.estimator.evaluate(eval_loader, self.training_model)
            self.recorder.tik_end(mode='eval')
            self.recorder.log_verbose_and_save(eval_metric_result, eval_loss, self.training_model)

            if self.recorder.early_stop:
                break

        self.recorder.report_best_res()
        # test model
        if self.do_test:
            test_metric_res = self.test_model(self.data_dict['test'])
            self.recorder.report_test_result(test_metric_res)


if __name__ == '__main__':
    pass
