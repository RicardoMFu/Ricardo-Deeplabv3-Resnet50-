from dataloader.dataset_tunnel import TunnelDataset

ds = TunnelDataset(root="data", split="train", img_size=(512, 512))

print("数据量:", len(ds))

img, mask = ds[0]

print("img:", img.shape)  # [3,512,512]
print("mask:", mask.shape)  # [512,512]
print("mask unique:", mask.unique())
for i in range(10):
    print(i, ds[i][1].unique())
