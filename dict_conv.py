import torch
import models.wideresnet as models
from collections import OrderedDict

dict = torch.load("./epoch=799-step=77600.ckpt")
st_dict = OrderedDict()
st_dict["conv1.weight"] = dict["state_dict"]["resnet_simsiam.backbone.0.weight"]
modelemb = models.build_wideresnet(28,2,dropout=0,num_classes=10)
for i in dict["state_dict"].keys():
    if(i[0:23]=="resnet_simsiam.backbone"):
        if(i[24]=="4"):
            st_dict["bn1."+i[26:]] = dict["state_dict"][i]
        elif(i[9]!="0"):
            st_dict["block"+i[24:]] = dict["state_dict"][i]


for i in st_dict.keys():
    print(i)

# print("\n\n hello \n\n")

for i in modelemb.state_dict().keys():
    if i in st_dict.keys():
        modelemb.state_dict()[i] = st_dict[i]

# # print("\n\n hello \n\n")

torch.save(modelemb.state_dict(),"./barlow_weights.pt")

# for i in st_dict.keys():
#     print(i)