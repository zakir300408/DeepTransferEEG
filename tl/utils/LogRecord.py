# -*- coding: utf-8 -*-
# @Time    : 2021/12/8 19:08
# @Author  : wenzhang
# @File    : LogRecord.py

import torch as tr
import os.path as osp
import os
import time
from datetime import datetime
from datetime import timedelta, timezone

from utils.utils import create_folder


class LogRecord:
    def __init__(self, args):
        self.args = args
        self.result_dir = args.result_dir
        try:
            self.data_env = 'gpu' if tr.cuda.get_device_name(0) != 'GeForce GTX 1660 Ti' else 'local'
        except Exception:
            self.data_env = 'local'
        self.data_name = args.data
        self.method = args.method
        self.align = args.align

    def log_init(self):
        os.makedirs(self.args.result_dir, exist_ok=True)  # ensure logs folder exists
        file_name_head = 'log_' + str(self.args.method) + '_' + str(self.args.data_name) + '_'
        time_str = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
        self.args.out_file = open(osp.join(self.args.result_dir, file_name_head + time_str + '.txt'), 'w')
        self.args.out_file.write(self._print_args() + '\n')
        self.args.out_file.flush()
        return self.args

    def record(self, log_str):
        self.args.out_file.write(log_str + '\n')
        self.args.out_file.flush()
        return self.args

    def _print_args(self):
        s = "==========================================\n"
        for arg, content in self.args.__dict__.items():
            s += "{}:{}\n".format(arg, content)
        return s
