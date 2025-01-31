# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================


import numpy as np
from torch.utils import data
from fuxictr.utils import load_h5
import torch
import logging
from tqdm import tqdm
from bitarray import bitarray

class Dataset(data.Dataset):
    def __init__(self, feature_map, data_path):
        self.feature_map = feature_map
        self.darray = self.load_data_array(data_path)
        
    def __getitem__(self, index):
        return self.darray[index, :]
    
    def __len__(self):
        return self.darray.shape[0]

    def load_data_array(self, data_path):
        data_dict = load_h5(data_path) # dict of arrays from h5
        data_arrays = []
        all_cols = list(self.feature_map.features.keys()) + self.feature_map.labels
        for col in all_cols:
            array = data_dict[col]
            if array.ndim == 1:
                data_arrays.append(array.reshape(-1, 1))
            else:
                data_arrays.append(array)
        data_tensor = torch.from_numpy(np.hstack(data_arrays))
        return data_tensor


class DataLoader(data.DataLoader):
    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False, num_workers=1, **kwargs):
        self.dataset = Dataset(feature_map, data_path)
        super(DataLoader, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=num_workers)
        self.num_samples = len(self.dataset)
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / self.batch_size))
        self.type = kwargs.get("type", "train")

        if self.type == "train":
            # Initialize feature_sets to store seen features during training
            self.feature_sets = {field: set() for field in self.dataset.feature_map.features.keys()}
            self.train_ratio = kwargs.get("train_ratio", 1)
            self.train_order = []

            self._make_batch_order()

            self.num_batches = len(self.batch_order)
            logging.info(f"[Selected Mode] Train ratio: {self.train_ratio} with {self.num_batches} batches.")

    def __len__(self):
        return self.num_batches

    def _check_coverage_old(self, batch, update = False):
        """Check if the current batch covers new values for any field."""
        new_values_count = 0

        for field in self.dataset.feature_map.features.keys():
            idx = self.dataset.feature_map.column_index[field]
            field_values = set(batch[:, idx].numpy())  # Convert batch values to a set

            # Get new values by subtracting already seen values
            new_values = field_values - self.feature_sets[field]

            if new_values:
                if update:
                    self.feature_sets[field].update(new_values)  # Add new values to feature_sets
                new_values_count += len(new_values)

        return new_values_count  # Return the count of new values covered

    def _check_coverage(self, batch_id, update = False):
        """Check if the current batch covers new values for any field."""
        new_values_count = 0

        for field in self.dataset.feature_map.features.keys():
            field_values = self.bitmap_dict[batch_id][field]
            total_values = self.total_dict[field]

            # Get new values by subtracting already seen values
            new_values = field_values & ~total_values

            if new_values:
                if update:
                    self.total_dict[field] |= new_values  # Add new values to feature_sets
                new_values_count += new_values.count()

        return new_values_count  # Return the count of new values covered

    def _prepare_bitmap_for_batch(self):
        batch_dicts = [{} for _ in range(self.num_batches)]
        total_dict = {}
        for field in tqdm(self.dataset.feature_map.features.keys(), desc="Processing fields", unit="field"):
            idx = self.dataset.feature_map.column_index[field]
            max_len = self.dataset.feature_map.features[field]['vocab_size']
            total_dict[field] = bitarray(max_len)
            total_dict[field].setall(0)
            for i, batch in enumerate(batch_dicts):
                bitmaps = bitarray(max_len)
                bitmaps.setall(0)
                field_values = set(self.dataset[:][i*self.batch_size: (i+1)*self.batch_size, idx].numpy().astype(int))

                for value in field_values:
                    bitmaps[value] = 1

                batch_dicts[i][field] = bitmaps
        return batch_dicts, total_dict

    def _make_batch_order(self):
        if self.type == "train":
            self.batch_order = []
            self.bitmap_dict, self.total_dict = self._prepare_bitmap_for_batch()
            # Step 1: Initialize all batch indices
            batch_indices = list(range(self.num_batches))

            while True:
                batch_coverages = []
                for batch_index in batch_indices:
                    new_values_count = self._check_coverage(batch_index)  # Get the new values covered by this batch
                    batch_coverages.append((batch_index, new_values_count))

                batch_coverages.sort(key=lambda x: x[1], reverse=True)

                best_batch_index, new_coverage = batch_coverages[0]

                if new_coverage == 0:
                    batch_coverages.pop(0)
                    logging.info(f"Yielding batch {best_batch_index}, which covers no new values.")
                    break

                dist = abs(len(self.batch_order) - self.num_batches * self.train_ratio)
                if int(dist) % 20 == 0:
                    logging.info(f"Dist to target: {dist} batches towards {self.num_batches * self.train_ratio}.")
                batch_coverages.pop(0)  # Remove the batch that is being yielded

                self._check_coverage(best_batch_index,
                                     update=True)  # Update feature_sets with new values covered by this batch

                # logging.info(f"Yielding batch {best_batch_index}, which covers new values of {new_coverage}.")
                self.batch_order.append(best_batch_index)

                # If all features are covered, exit the loop
                if len(batch_coverages) == 0:
                    break

                if len(self.batch_order) >= self.num_batches * self.train_ratio:
                    break
    def __iter__(self):
        if self.type != "train":
            for batch in super(DataLoader, self).__iter__():
                yield batch
        else:
            assert hasattr(self, 'batch_order'), "Batch order not initialized."
            for i in self.batch_order:
                yield self.dataset[i*self.batch_size: (i+1)*self.batch_size]



class H5SelectedDataLoader(object):
    def __init__(self, feature_map, stage="both", train_data=None, valid_data=None, test_data=None,
                 batch_size=32, shuffle=True, **kwargs):
        logging.info("Loading data...")
        train_gen = None
        valid_gen = None
        test_gen = None
        self.stage = stage
        if stage in ["both", "train"]:
            train_gen = DataLoader(feature_map, train_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
            logging.info("Train samples: total/{:d}, blocks/{:d}".format(train_gen.num_samples, 1))     
            valid_gen = DataLoader(feature_map, valid_data, batch_size=batch_size, shuffle=False, type = "valid", **kwargs)
            logging.info("Validation samples: total/{:d}, blocks/{:d}".format(valid_gen.num_samples, 1))
        if stage in ["both", "test"]:
            test_gen = DataLoader(feature_map, test_data, batch_size=batch_size, shuffle=False, type = "test",  **kwargs)
            logging.info("Test samples: total/{:d}, blocks/{:d}".format(test_gen.num_samples, 1))
        self.train_gen, self.valid_gen, self.test_gen = train_gen, valid_gen, test_gen

    def make_iterator(self):
        if self.stage == "train":
            logging.info("Loading train and validation data done.")
            return self.train_gen, self.valid_gen
        elif self.stage == "test":
            logging.info("Loading test data done.")
            return self.test_gen
        else:
            logging.info("Loading data done.")
            return self.train_gen, self.valid_gen, self.test_gen
