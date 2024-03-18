import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


dataPath = r'C:\Users\FELIX SAM(TECH WATT)\Desktop\data'

tranform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Resize((224,224)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomVerticalFlip(),
  transforms.Pad(20,fill=1),
  transforms.RandomRotation(10),
  transforms.Normalize(mean=[0.5,0.6,0.4],std=[0.5,0.4,0.7])


])

data = ImageFolder(dataPath,transform = tranform)
data_loader = DataLoader(data,shuffle=True,batch_size=2)
for images,labels in data_loader:
    for img in images:
        img = img.transpose(0,2)
        print(img.shape)
        plt.imshow(img)
        plt.show()
    # print(images.shape)