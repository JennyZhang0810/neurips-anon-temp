import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.ytb_vos import YoutubeVOSDataset
from datasets.ytb_vis import YoutubeVISDataset
from datasets.mydataset import MyDataset
from datasets.saliency_modular import SaliencyDataset
from datasets.vipseg import VIPSegDataset
from datasets.mvimagenet import MVImageNetDataset
from datasets.sam import SAMDataset
from datasets.uvo import UVODataset
from datasets.uvo_val import UVOValDataset
from datasets.mose import MoseDataset

from datasets.vitonhd import VitonHDDataset
from datasets.fashiontryon import FashionTryonDataset
from datasets.lvis import LvisDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from torch.utils.data import ConcatDataset
from cldm.hack import disable_verbosity, enable_sliced_attention
from omegaconf import OmegaConf

save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()

# Configs
resume_path = '/data5/zhangjiening/AnyDoor-main/lightning_logs/version_44 (1)/checkpoints/epoch=271-step=98735.ckpt'
batch_size = 2
logger_freq = 500
learning_rate = 1e-6
sd_locked = False
only_mid_control = False
n_gpus = 1
accumulate_grad_batches=2

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./configs/anydoor.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Datasets
# DConf = OmegaConf.load('./configs/datasets.yaml')
# dataset1 = YoutubeVOSDataset(**DConf.Train.YoutubeVOS)  
# dataset2 =  SaliencyDataset(**DConf.Train.Saliency) 
# dataset3 = VIPSegDataset(**DConf.Train.VIPSeg) 
# dataset4 = YoutubeVISDataset(**DConf.Train.YoutubeVIS) 
# dataset5 = MVImageNetDataset(**DConf.Train.MVImageNet)
# dataset6 = SAMDataset(**DConf.Train.SAM)
# dataset7 = UVODataset(**DConf.Train.UVO.train)
# dataset8 = VitonHDDataset(**DConf.Train.VitonHD)
# dataset9 = UVOValDataset(**DConf.Train.UVO.val)
# dataset10 = MoseDataset(**DConf.Train.Mose)
# dataset11 = FashionTryonDataset(**DConf.Train.FashionTryon)
# dataset12 = LvisDataset(**DConf.Train.Lvis)

# image_data = [dataset2, dataset6, dataset12]
# video_data = [dataset1, dataset3, dataset4, dataset7, dataset9, dataset10 ]
# tryon_data = [dataset8, dataset11]
# threed_data = [dataset5]

DConf = OmegaConf.load('./configs/my_datasets.yaml')
# dataset1 = MyDataset(**DConf.Train.rs508_cv1)
# dataset2 = MyDataset(**DConf.Train.rs508_cv2)
# dataset3 = MyDataset(**DConf.Train.rs508_dd1)
# dataset4 = MyDataset(**DConf.Train.rs508_dd2)
# image_data = [dataset1, dataset2, dataset3, dataset4]
# image_data = [dataset1, dataset2, dataset4]
image_data = MyDataset(**DConf.Train.MIG_Sea)



# The ratio of each dataset is adjusted by setting the __len__ 
# dataset = ConcatDataset( image_data + video_data + tryon_data +  threed_data + video_data + tryon_data +  threed_data  )
# dataset = ConcatDataset(image_data)
dataset = image_data
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=n_gpus, strategy="ddp", precision=16, accelerator="gpu", callbacks=[logger], progress_bar_refresh_rate=1, accumulate_grad_batches=accumulate_grad_batches)

# Train!
trainer.fit(model, dataloader)
