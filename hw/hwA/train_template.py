#First, download medical_mnist.zip
#from Kaggle https://www.kaggle.com/datasets/andrewmvd/medical-mnist
#import all the helper functions

from utils_mnist import *
import numpy as np



class Trainer:
    def __init__(
        self,
        args,
        train_loader,
        validation_loader,
        model,
        loss_function,
        optimizer
    ):
        """
        Store arguments, data loaders, model,
        loss function, optimizer, and device.
        """

        self.args = args
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.loss_function = loss_function
        self.optimizer = optimizer

        # Use GPU if available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        print('Using device:', self.device)

        # Move model to CPU/GPU
        self.model = model.to(self.device)

    def overall_loop(self):
        """
        Run training for specified epochs,
        print train/validation loss each epoch.
        """

        for epoch in range(self.args['num_epochs']):

            train_losses = self.training_loop(
                self.train_loader
            )

            validation_losses = self.validation_loop(
                self.validation_loader
            )

            train_loss = np.mean(train_losses)
            validation_loss = np.mean(validation_losses)

            print(
                'Epoch [{}/{}], '
                'Train Loss: {:.4f}, '
                'Validation Loss: {:.4f}'.format(
                    epoch + 1,
                    self.args['num_epochs'],
                    train_loss,
                    validation_loss
                )
            )

    def training_loop(self, train_loader):
        """
        Iterate through training batches
        and return list of batch losses.
        """

        # Enable training mode
        self.model.train()

        losses = []

        for train_batch in train_loader:

            loss = self.training_step(train_batch)

            losses.append(loss)

        return losses

    def validation_loop(self, validation_loader):
        """
        Iterate through validation batches
        and return list of batch losses.
        """

        # Disable dropout etc.
        self.model.eval()

        losses = []

        # Validation does not need gradient
        with torch.no_grad():

            for validation_batch in validation_loader:

                loss = self.validation_step(
                    validation_batch
                )

                losses.append(loss)

        return losses

    def training_step(self, train_batch):
        """
        Forward pass
        -> compute loss
        -> backpropagation
        -> update weights
        """

        images, labels = train_batch

        # Move data to device
        images = images.to(
            self.device,
            dtype=torch.float32
        )

        labels = labels.to(
            self.device,
            dtype=torch.long
        )

        # Logistic regression requires flattened input
        if self.args['model'] == 'logistic_regression':
            images = images.reshape(
                -1,
                self.args['img_size']
            )

        # Clear previous gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(images)

        # Compute loss
        loss = self.loss_function(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update model weights
        self.optimizer.step()

        # Convert tensor loss to Python number
        return loss.item()

    def validation_step(self, validation_batch):
        """
        Forward pass and compute validation loss.
        No backpropagation.
        """

        images, labels = validation_batch

        images = images.to(
            self.device,
            dtype=torch.float32
        )

        labels = labels.to(
            self.device,
            dtype=torch.long
        )

        if self.args['model'] == 'logistic_regression':
            images = images.reshape(
                -1,
                self.args['img_size']
            )

        # Forward pass
        outputs = self.model(images)

        # Compute validation loss
        loss = self.loss_function(
            outputs,
            labels
        )

        return loss.item()


# arguments
def get_arguments():
    # Hyper-parameters
    args = {'img_size': 64 * 64,
            'num_classes': 6,
            'num_epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.001,
            'model': 'logistic_regression'} # MLP or logistic_regression
    return args


args = get_arguments()
print(args)

# Medical MNIST dataset (images and labels)
train_loader, validation_loader = get_medical_mnist(args=args)
print('Done with data preparation')

print('Length of train dataset:')
print(len(train_loader.dataset))
print('Length of validation dataset:')
print(len(validation_loader.dataset))
print('Length of train dataloader:')
print(len(train_loader))
print('Length of validation dataloader:')
print(len(validation_loader))

some_index = np.random.randint(0, len(train_loader.dataset), 10)
some_imgs = [train_loader.dataset.__getitem__(idx)[0] for idx in some_index]
some_labels = [train_loader.dataset.__getitem__(idx)[1] for idx in some_index]

show_examples(some_imgs[:5], some_labels[:5])

# get model
if args['model'] == 'logistic_regression':
    print('Using logistic regression')
    model = nn.Linear(args['img_size'], args['num_classes'])
elif args['model'] == 'MLP':
    print('MLP')
    model = MLP(dropout=0, hidden_1=512, hidden_2=512)

# Loss and optimizer
loss_function = nn.CrossEntropyLoss() # this combined the LogSoftmax and NLLLoss
optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])

trainer = Trainer(args, train_loader, validation_loader, model, loss_function, optimizer)

trainer.overall_loop()

correct = 0
total = 0
for images, labels in validation_loader:
    
    images = images.to(
        trainer.device,
        dtype=torch.float32
    )

    labels = labels.to(
        trainer.device,
        dtype=torch.long
    )
    
    images = images.reshape(-1, args['img_size'])
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(correct)
print(total)

print('Accuracy of the model on the {} validation images: {:.2f} %'.format(total, 100 * correct / total))
