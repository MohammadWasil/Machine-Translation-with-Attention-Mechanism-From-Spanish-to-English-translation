
class Trainer:
    def __init__(self, model, config, device, embedding_size_english):
        import torch
        import torch.nn as nn
        from torch.optim import Adam

        self.model = model
        self.config = config
        self.device = device
        self.EMBEDDING_SIZE_ENGLISH = embedding_size_english

    def evaluate(self, iterator, loss_function):

        self.model.eval()
        
        epoch_loss = 0
        
        with torch.no_grad():
        
            for i, batch in enumerate(iterator):

                source = batch.src
                target = batch.trg

                # feed the source and target sentence into the model to get prediction
                output = self.model(source, target, 0)
                
                prediction = output[1:].reshape(-1, self.EMBEDDING_SIZE_ENGLISH)
                # shape: [sequence_len_target-1*BATCH_SIZE, EMBEDDING_SIZE_ENGLISH]        
                
                actual = target[1:].reshape(-1)
                # shape: [sequence_len_target-1*BATCH_SIZE]

                loss = loss_function(prediction, actual)

                epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def train(self, train_iterator, valid_iterator):
        
        optimizer = Adam(self.model.parameters(), self.config['lr'])
        loss_function = nn.CrossEntropyLoss()
        
        for epoch in range(self.config["epochs"]):
            
            self.model.train()
            self.model.to(self.device)
            train_loss = 0
            for i, batch in enumerate(train_iterator):
            
                source = batch.src
                target = batch.trg
                # src shape: [sequence_len_source, BATCH_SIZE]
                # trg shape: [sequence_len_target, BATCH_SIZE]
        
                optimizer.zero_grad()
                
                # feed the source and target sentence into the model to get prediction
                output = self.model(source, target)
                #output shape : [sequence_len_target, BATCH_SIZE, EMBEDDING_SIZE_ENGLISH]
        
                prediction = output[1:].reshape(-1, self.EMBEDDING_SIZE_ENGLISH)
                # prediction shape: [sequence_len_target-1*BATCH_SIZE, EMBEDDING_SIZE_ENGLISH]        
        
                actual = target[1:].reshape(-1)
                # actual shape: [sequence_len_target-1*BATCH_SIZE]
                
                loss = loss_function(prediction, actual)
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                
                optimizer.step()
                
                train_loss += loss.item()
            train_loss = train_loss / len(train_iterator)
            
            valid_loss = self.evaluate(self.model, valid_iterator, loss_function)

            # save the model at every epoch.
            state = {
                    'epoch': epoch+1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict()
            }
            torch.save(state, f'model_{epoch+1}.pickle')

            print(f'Epoch: {epoch+1} | Train Loss: {train_loss} | Val. Loss: {valid_loss}')   