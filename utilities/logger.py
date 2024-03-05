import csv
import datetime
import os
import pickle as pkl
import numpy as np
from utilities.misc import gimme_save_string


class CSV_Writer():
    def __init__(self, save_path):
        self.save_path = save_path
        self.written = []
        self.n_written_lines = {}

    def log(self, group, segments, content):
        if group not in self.n_written_lines.keys():
            self.n_written_lines[group] = 0

        with open(self.save_path + '_' + group + '.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            if group not in self.written: writer.writerow(segments)
            for line in content:
                writer.writerow(line)
                self.n_written_lines[group] += 1

        self.written.append(group)

def set_logging(opt):
    checkfolder = opt.save_path + '/' + opt.save_name
    if opt.save_name == '':
        date = datetime.datetime.now()
        time_string = '{}-{}-{}-{}-{}-{}'.format(date.year, date.month,
                                                 date.day, date.hour,
                                                 date.minute, date.second)
        checkfolder = opt.save_path + '/{}_{}_'.format(
            opt.dataset.upper(), opt.arch.upper()) + time_string
    counter = 1
    while os.path.exists(checkfolder):
        checkfolder = opt.save_path + '/' + opt.save_name + '_' + str(counter)
        counter += 1
    os.makedirs(checkfolder)
    opt.save_path = checkfolder

    save_opt = opt

    with open(save_opt.save_path + '/Parameter_Info.txt', 'w') as f:
        f.write(gimme_save_string(save_opt))
    pkl.dump(save_opt, open(save_opt.save_path + "/hypa.pkl", "wb"))


class Progress_Saver():
    def __init__(self):
        self.groups = {}

    def log(self, segment, content, group=None):
        if group is None: group = segment
        if group not in self.groups.keys():
            self.groups[group] = {}

        if segment not in self.groups[group].keys():
            self.groups[group][segment] = {'content': [], 'saved_idx': 0}

        self.groups[group][segment]['content'].append(content)


class LOGGER():
    def __init__(self,
                 opt,
                 sub_loggers=[],
                 prefix=None,
                 start_new=True):

        self.prop = opt
        self.prefix = '{}_'.format(prefix) if prefix is not None else ''
        self.sub_loggers = sub_loggers

        ### Make Logging Directories
        if start_new: set_logging(opt)

        ### Set Graph and CSV writer
        self.csv_writer, self.progress_saver = {}, {}
        for sub_logger in sub_loggers:
            csv_savepath = opt.save_path + '/CSV_Logs'
            if not os.path.exists(csv_savepath): os.makedirs(csv_savepath)
            self.csv_writer[sub_logger] = CSV_Writer(
                csv_savepath + '/Data_{}{}'.format(self.prefix, sub_logger))

            prgs_savepath = opt.save_path + '/Progression_Plots'
            if not os.path.exists(prgs_savepath): os.makedirs(prgs_savepath)

            self.progress_saver[sub_logger] = Progress_Saver()

        ### WandB Init
        self.save_path = opt.save_path

    def update(self, *sub_loggers, all=False):
        online_content = []

        if all: sub_loggers = self.sub_loggers

        for sub_logger in list(sub_loggers):
            for group in self.progress_saver[sub_logger].groups.keys():
                pgs = self.progress_saver[sub_logger].groups[group]
                segments = pgs.keys()
                per_seg_saved_idxs = [
                    pgs[segment]['saved_idx'] for segment in segments
                ]
                per_seg_contents = [
                    pgs[segment]['content'][idx:]
                    for segment, idx in zip(segments, per_seg_saved_idxs)
                ]
                per_seg_contents_all = [
                    pgs[segment]['content']
                    for segment, idx in zip(segments, per_seg_saved_idxs)
                ]

                #Adjust indexes
                for content, segment in zip(per_seg_contents, segments):
                    self.progress_saver[sub_logger].groups[group][segment][
                        'saved_idx'] += len(content)

                tupled_seg_content = [
                    list(seg_content_slice)
                    for seg_content_slice in zip(*per_seg_contents)
                ]

                self.csv_writer[sub_logger].log(group, segments,
                                                tupled_seg_content)

                for i, segment in enumerate(segments):
                    if group == segment:
                        name = sub_logger + ': ' + group
                    else:
                        name = sub_logger + ': ' + group + ': ' + segment
                    online_content.append((name, per_seg_contents[i]))

