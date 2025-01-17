import torch
import torch.utils.data.dataloader
from torch.utils.data.dataloader import default_collate

class BasePredictor():
    def __init__(self,
                 batch_size=128,
                 max_epochs=20,
                 valid=None,
                 device=None,
                 metrics={},
                 learning_rate=1e-3,
                 max_iters_in_epoch=1e20,
                 grad_accumulate_steps=1):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.valid = valid
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.max_iters_in_epoch = max_iters_in_epoch
        self.grad_accumulate_steps = grad_accumulate_steps

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        self.epoch = 0

    def fit_dataset(self, data, valid, collate_fn=default_collate, callbacks=[]):

        # Start the training loop.
        while self.epoch < self.max_epochs:
            # train and evaluate train score
            print('training %i' % self.epoch)
            # TODO: create dataloader for `train`.
            # You can set batch_size as `self.batch_size` here,
            # and `collate_fn=collate_fn`.
            dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
             collate_fn=collate_fn, shuffle=True)

            # train epoch
            log_train = self._run_epoch(dataloader, True)

            # evaluate valid score
            if self.valid is not None:
                print('evaluating %i' % self.epoch)
                # TODO: create dataloader for `valid`.
                # You can set batch_size as `self.batch_size` here,
                # and `collate_fn=collate_fn`.

                dataloader = torch.utils.data.DataLoader(valid, batch_size=self.batch_size, 
                    collate_fn=valid.collate_fn, shuffle=True)
                
                # evaluate model
                log_valid = self._run_epoch(dataloader, True)
            else:
                log_valid = None

            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)

            self.epoch += 1

    def predict_dataset(self, data,
                        collate_fn=default_collate,
                        batch_size=None,
                        predict_fn=None):
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        # set model to eval mode
        self.model.eval()
        # make dataloader
        # TODO: create dataloader for `data`.
        # You can set batch_size as `self.batch_size` here,
        # and `collate_fn=collate_fn`.
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size,
        	collate_fn=collate_fn, shuffle=False)

        # evaluate model
        ys_ = []
        with torch.no_grad():
            for batch in dataloader:
                batch_y_ = predict_fn(batch)
                ys_.append(batch_y_)

        ys_ = torch.cat(ys_, 0)

        return ys_

    def save(self, path):
        torch.save({
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        # TODO: Load saved model from path here.
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _run_epoch(self, dataloader, training):
        # set model training/evaluation mode
        self.model.train(training)

        # run batches for train
        loss = 0

        # reset metric accumulators
        for metric in self.metrics:
            metric.reset()

        if training:
            iter_in_epoch = min(len(dataloader), self.max_iters_in_epoch)
            description = 'training'
        else:
            iter_in_epoch = len(dataloader)
            description = 'evaluating'

        # run batches
        for i, batch in enumerate(dataloader):
            if training and i >= iter_in_epoch:
                break

            if training:
                #print('training', batch['labels'].size())
                output, batch_loss = \
                    self._run_iter(batch, training)

                batch_loss /= self.grad_accumulate_steps

                # accumulate gradient - zero_grad
                if i % self.grad_accumulate_steps == 0:
                    # TODO: call zero gradient here
                    self.optimizer.zero_grad()

                # TODO: Call backward on `batch_loss` here.
                batch_loss.backward()

                # accumulate gradient - step
                if (i + 1) % self.grad_accumulate_steps == 0:
                    # TODO: update gradient here
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    #print('valid', batch['labels'].size())
                    output, batch_loss = \
                        self._run_iter(batch, training)

            # accumulate loss and metric scores
            loss += batch_loss.item()
            for metric in self.metrics:
                metric.update(output, batch)

            if i % 10 == 0:
                print('{0}: {1}/{2} [loss: {3}, {4}: {5}]'.format(description, i, iter_in_epoch,
                 '%.2f' % (loss/(i+1)), self.metrics[0].name, self.metrics[0].print_score()))

        # calculate averate loss and metrics
        loss /= iter_in_epoch

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        for metric in self.metrics:
            score = metric.get_score()
            print('{}: {} '.format(metric.name, score))
            epoch_log[metric.name] = score

        print('loss=%f\n' % loss)
        return epoch_log

    def _run_iter(self, batch, training):
        """ Run iteration for training.

        Args:
            batch (dict)
            training (bool)

        Returns:
            predicts: Prediction of the batch.
            loss (FloatTensor): Loss of the batch.
        """
        pass

    def _predict_batch(self, batch):
        """ Run iteration for predicting.

        Args:
            batch (dict)

        Returns:
            predicts: Prediction of the batch.
        """
        pass
