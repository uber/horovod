# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
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
# ==============================================================================

from pytorch_lightning import LightningModule

from horovod.spark.common.util import to_list


def to_lightning_module(model, optimizer, loss_fns, loss_weights, feature_cols, label_cols, sample_weights_col):
    loss_weights = loss_weights or [1.0 / len(label_cols)] * len(label_cols)
    loss_fns = to_list(loss_fns, len(label_cols))

    class _EstimatorLightningModule(LightningModule):
        def __init__(self):
            super().__init__()

        def forward(self, **kwargs):
            return model(**kwargs)

        def configure_optimizers(self):
            return optimizer

        def training_step(self, batch, batch_nb):
            loss = self._step(batch)
            tensorboard_logs = {'train_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}

        def validation_step(self, batch, batch_nb):
            loss = self._step(batch)
            tensorboard_logs = {'val_loss': loss}
            return {'loss': loss, 'log': tensorboard_logs}

        def _step(self, batch):
            inputs = {feature: batch[feature] for feature in feature_cols}
            labels = [batch[label] for label in label_cols]
            sample_weights = batch[sample_weights_col] if sample_weights_col else None
            outputs = self(**inputs)
            outputs, labels = self._transform_outputs(outputs, labels)
            return self._calculate_loss(outputs, labels, sample_weights)

        def _transform_outputs(self, outputs, labels):
            if type(outputs) != tuple and type(outputs) != list:
                outputs = [outputs]

            # reshape labels to match the output shape of the model
            if hasattr(outputs[0], 'shape'):
                labels = [label.reshape(output.shape)
                          if output.shape.numel() == label.shape.numel() else label
                          for label, output in zip(labels, outputs)]
            return outputs, labels

        def _calculate_loss(self, outputs, labels, sample_weights=None):
            if sample_weights is not None:
                # when reduction='none', loss function returns the value of all the losses
                # from all the samples. We multiply each sample's weight to its loss and
                # then take the mean of the weight adjusted losses from all the samples in the
                # batch. Note that this approach is not "weighted average" because the sum of
                # the sample weights in each batch does not necessarily add up to one. If we add
                # the weights and divide the sum to the sum of weights, the impact of two
                # samples with identical weights but in different batches will not be equal on
                # the calculated gradients.
                losses = []
                for output, label, loss_fn, loss_weight in zip(outputs, labels,
                                                               loss_fns, loss_weights):
                    weight_adjusted_sample_losses = \
                        loss_fn(output, label, reduction='none').flatten() * sample_weights
                    output_loss = weight_adjusted_sample_losses.mean()
                    losses.append(output_loss * loss_weight)
            else:
                losses = [loss_fn(output, label) * loss_weight for
                          output, label, loss_fn, loss_weight in
                          zip(outputs, labels, loss_fns, loss_weights)]

            loss = sum(losses)
            return loss

    return _EstimatorLightningModule()
