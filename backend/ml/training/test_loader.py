from dataset import get_data_loaders

data_dir = "../data/chest_xray"

train_loader, val_loader, test_loader, train_dataset = get_data_loaders(data_dir)

print("Classes:", train_dataset.classes)
print("Total training samples:", len(train_dataset))

for images, labels in train_loader:
    print("Batch image shape:", images.shape)
    print("Batch label shape:", labels.shape)
    break
