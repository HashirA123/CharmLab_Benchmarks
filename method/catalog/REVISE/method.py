
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from data.data_object import DataObject
from evaluation.utils import check_counterfactuals
from evaluation.utils import check_counterfactuals
from experiment_utils import deep_merge, reconstruct_encoding_constraints
from method.method_factory import register_method
from method.method_object import MethodObject
from model.catalog.autoencoder.vae import VariationalAutoencoder
from model.model_object import ModelObject
import logging

@register_method("REVISE")
class Revise(MethodObject):
    def __init__(self, data: DataObject, model: ModelObject, vae = None, config_override = None):
        super().__init__(data, model, config_override)

        # get configs from config file
        self.config = yaml.safe_load(open("method/catalog/REVISE/library/config.yml", 'r'))
        
        # merge configs with user specified, if they exist
        if self._config_override is not None:
            self.config = deep_merge(self.config, self._config_override)

        self._feature_order = self._data.get_feature_names(expanded=True)

        self._target_column = self._data.get_target_column()
        self._lambda = self.config['lambda']
        self._optimizer = self.config['optimizer']
        self._lr = self.config['lr']
        self._max_iter = self.config['max_iter']
        self._target_class = self.config['target_class']
        self._binary_cat_features = self.config['binary_cat_features']

        vae_params = self.config['vae_params']

        print([int(sum(self._model.get_mutable_mask()))] + vae_params['layers'])

        self.vae = (
            vae 
            if vae
            else VariationalAutoencoder(
                data_name = self.config['data_name'] if self.config['data_name'] else "Temp",
                layers=[int(sum(self._model.get_mutable_mask()))] + vae_params['layers'],
                mutable_mask=self._model.get_mutable_mask(),
            )
        )
        if vae_params['train']:
            self.vae.fit(
                xtrain=self._model.get_train_data()[0],
                lambda_reg=vae_params['lambda_reg'],
                epochs=vae_params['epochs'],
                lr=vae_params['lr'],
                batch_size=vae_params['batch_size'],
            )
        else:
            try:
                self.vae.load(self._data.get_processed_data().shape[1] - 1)
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Loading of Autoencoder failed. {}".format(str(exc))
                )
            
    def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        factuals = factuals[self._feature_order] # ensure the feature ordering is correct for the model input

        # pay attention to categorical features
        encoded_feature_names = self._data.get_categorical_features(expanded=True)

        cat_features_indices = []
        for features in encoded_feature_names:
            # Find the indices of these encoded features in the processed dataframe
            indices = [factuals.columns.get_loc(feat) for feat in features]
            cat_features_indices.append(indices)

        list_cfs = self._counterfactual_optimization(
            cat_features_indices, device, factuals
        )

        list_cfs = np.array(list_cfs)
        df_cfs = pd.DataFrame(list_cfs, columns=self._feature_order) # ensure the feature ordering is correct for the model input
        df_cfs = check_counterfactuals(self._model, self._data, df_cfs, factuals.index) 
        return df_cfs

    def _counterfactual_optimization(self, cat_features_indices, device, df_fact):
        # prepare data for optimization steps
        test_loader = torch.utils.data.DataLoader(
            df_fact.values, batch_size=1, shuffle=False
        )
        mutable_mask_tensor = torch.tensor(
            self.vae.mutable_mask, dtype=torch.bool, device=device
        )
        mutable_indices = torch.nonzero(mutable_mask_tensor, as_tuple=False).squeeze(1)
        mutable_indices = (
            mutable_indices if mutable_indices.ndim else mutable_indices.unsqueeze(0)
        )

        list_cfs = []
        for query_instance in test_loader:
            query_instance = query_instance.float().to(device)

            target = torch.FloatTensor(self._target_class).to(device)
            target_prediction = np.argmax(np.array(self._target_class))

            # encode the mutable features
            z = self.vae.encode(query_instance[:, self.vae.mutable_mask])[0]
            # add the immutable features to the latents
            z = torch.cat([z, query_instance[:, ~self.vae.mutable_mask]], dim=-1)
            z = z.clone().detach().requires_grad_(True)

            if self._optimizer == "adam":
                optim = torch.optim.Adam([z], self._lr)
                # z.requires_grad = True
            else:
                optim = torch.optim.RMSprop([z], self._lr)

            candidate_counterfactuals = []  # all possible counterfactuals
            # distance of the possible counterfactuals from the intial value -
            # considering distance as the loss function (can even change it just the distance)
            candidate_distances = []
            all_loss = []

            for idx in range(self._max_iter):
                decoded_cf = self.vae.decode(z)

                index = mutable_indices.unsqueeze(0).expand(query_instance.size(0), -1)
                cf = query_instance.scatter(1, index, decoded_cf)

                cf_soft, cf_hard = (
                    cf,
                    reconstruct_encoding_constraints(
                        cf, cat_features_indices
                    ),
                )

                # output_soft = self._model(cf_soft)[0]
                output_soft = self._model.predict_proba(cf_soft).squeeze()
                output_hard = self._model.predict(cf_hard)[0][0]
                predicted = output_hard
                # _, predicted = torch.max(output_hard, 0)

                z.requires_grad = True
                loss = self._compute_loss(output_soft, cf_soft, query_instance, target)
                all_loss.append(loss)

                if predicted == target_prediction:
                    candidate_counterfactuals.append(
                        cf_hard.cpu().detach().numpy().squeeze(axis=0)
                    )
                    candidate_distances.append(loss.cpu().detach().numpy())

                loss.backward()
                optim.step()
                optim.zero_grad()
                cf.detach_()

            # Choose the nearest counterfactual
            if len(candidate_counterfactuals):
                logging.info("Counterfactual found!")
                array_counterfactuals = np.array(candidate_counterfactuals)
                array_distances = np.array(candidate_distances)

                index = np.argmin(array_distances)
                cf_tensor = (
                    torch.tensor(array_counterfactuals[index])
                    .unsqueeze(0)
                    .to(device)
                    .float()
                )
                cf_tensor = reconstruct_encoding_constraints(
                    cf_tensor,
                    cat_features_indices,
                    # self.config["binary_cat_features"],
                )
                list_cfs.append(cf_tensor.cpu().detach().numpy().squeeze(axis=0))
            else:
                logging.info("No counterfactual found")
                cf_tensor = reconstruct_encoding_constraints(
                    query_instance.clone(),
                    cat_features_indices,
                    # self.config["binary_cat_features"],
                )
                list_cfs.append(cf_tensor.cpu().detach().numpy().squeeze(axis=0))
        return list_cfs

    def _compute_loss(self, output, cf_initialize, query_instance, target):
        loss_function = nn.BCELoss()

        # classification loss
        # print(f"Here is the output {output} and the target {target}")
        loss1 = loss_function(output, target)
        # distance loss
        loss2 = torch.norm((cf_initialize - query_instance), 1)

        return loss1 + self._lambda * loss2

        
