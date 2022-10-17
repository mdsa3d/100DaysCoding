import torch
from PIL import Image
from torch import nn, save, load
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# download MNIST dataset, calssification data, 10 calsses 0 to 10
# root -> where you want to download
# download -> if data needs to be downloaded
# transform -> data transformation
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
# create datasets
dataset = DataLoader(train, 32) # 32 images

# create NN class
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )
            
    def forward(self, x):
        return self.model(x)

# instance of NN, loss, optimizer

clf = ImageClassifier().to('cpu') 
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    for epoch in range(10):
        for batch in dataset:
            X,y = batch
            X,y = X.to('cpu'), y.to('cpu')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backpropogation
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch} loss is {loss.item()}")
    
    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)
    with open('model_state.pt', 'wb') as f:
        clf.load_state_dict(load(f))
    
    img = Image.open('img_3.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cpu')

    print(torch.argmax(clf(img_tensor)))