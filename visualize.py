import matplotlib.pyplot as plt
import torchvision

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def visualizeData(dataloader, num_images, classes):
	# get some random training images
	if num_images > len(dataloader):
		num_images = len(dataloader)
	dataiter = iter(dataloader)
	images, labels = next(dataiter)
	images = images[0:num_images]

	# show images
	imshow(torchvision.utils.make_grid(images))
	# print labels
	print(' '.join(f'{classes[labels[j]]:5s}' for j in range(num_images)))

def drawLossAccuracyPlots(train_losses, train_accs, test_losses, test_accs):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accs)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accs)
    axs[1, 1].set_title("Test Accuracy")
    
plt.figure(figsize=(10,10))
plt.tight_layout()

right = 0
mistake = 0

def imshowready(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

def showIncorrectPreds(numImages, images, incorrectPreds, nonMatchingLabels):
    for i in range(numImages):
        image = images[i]
        pred = classes[incorrectPreds[i]]
        gt = classes[nonMatchingLabels[i]]

        plt.subplot(2,int(numImages/2),i+1)
        plt.imshow(imshowready(image))
        plt.axis('on')

        # ret = model.predict(data, batch_size=1)
        #print(ret)

        plt.title("Pred:" + pred + "\nGT:" + gt, color='#ff0000', fontdict={'fontsize': 12})

    plt.show()