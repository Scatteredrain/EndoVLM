import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import json
import yaml
import random
from dataset.custom_transforms import *
from torch.utils.data import DataLoader
import time
from PIL import Image, ImageFile
from tqdm import tqdm


def check_image_ok(path: str, use_pil_decode: bool = True):
    """
    Check whether an image file is readable (and not truncated/corrupted).

    Returns:
        ok (bool), msg (str)
    """
    if not isinstance(path, str) or len(path) == 0:
        return False, "empty path"
    if not os.path.exists(path):
        return False, "file not exists"
    if os.path.isdir(path):
        return False, "path is a directory"
    if os.path.getsize(path) == 0:
        return False, "file size is 0"

    # PIL: quick header check
    try:
        with Image.open(path) as im:
            im.verify()  # verifies header integrity (no full decode)
    except Exception as e:
        return False, f"PIL verify failed: {e}"

    if not use_pil_decode:
        return True, "ok"

    # PIL: force full decode to catch truncated/corrupted data
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.load()  # full decode
    except Exception as e:
        return False, f"PIL decode failed: {e}"

    return True, "ok"


def read_path_text_from_json(file_paths):
    """
    Read path and text from json file.
    """
    # file_path1 = '/mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_szy_0126_lim7_Final_addValidImageCount.jsonl'
    # file_path2 = '/mnt/data/xujianwei.xjw/胃镜/total/dataset/Image_Text_wzzx_0126_lim7_Final_addValidImageCount.jsonl'

    if isinstance(file_paths, str):
        file_paths = [file_paths]
    data_list = []
    for file_path in file_paths:
        assert os.path.exists(file_path), f"file {file_path} not exists"
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data_list.append(json.loads(line))
    return data_list

def load_config(file_path='config.yaml'):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)['transform_list']

class EndoscopyDataset(data.Dataset):
    def __init__(self, data_file, transform_list_file, tokenizer, is_train=True, testsize=224, max_imgs=1000, debug=False, debug_datasize=100, mode='clip'):
        data_list = read_path_text_from_json(data_file)
        self.is_train = is_train
        self.tokenizer = tokenizer 
        self.max_imgs = max_imgs 
        self.mode = mode

        if debug:
            data_list = random.sample(data_list, debug_datasize)

        self.bags = []
        self.texts = []
        self.normal_flags_all = []
        self.folder_names = []
        self.anatomy_flags_all = []
        self.image_nums = 0
        
        for i, content in enumerate(data_list):
            # if i % (len(data_list) // 10+1) == 0:
            #     print(f"{i}/{len(data_list)}")
            folder_path = content['image_path']
            parts = folder_path.rstrip('/').split('/')
            folder = parts[-2] + '/' + parts[-1]

            # images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            #          if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '报告' not in f]
            images = [os.path.join(folder_path, f) for f in content['valid_image_paths']]
            report = content['endo_report']
            abnormal_flags = content['site_abnormality_flags']
            normal_flags = [1-x for x in abnormal_flags]
            sub_reports_len = len(report.split('\n')[1:])
            # ok, msg = check_image_ok(images[0])
            # if not ok: 
            #     print(msg)
            #     continue
            if len(images) == 0: continue
            if len(normal_flags) != sub_reports_len:
                print(f"Error: flag length {len(normal_flags)} != sub_reports length {sub_reports_len}")
                print(report,'\n',abnormal_flags)
                continue

            modality = content['modality']
            anatomy_flags = [0,1,2,3,4,5,6,7] if (modality == 'gastroscopy') else [8,9,10,11,12,13,14,15,16]
            self.anatomy_flags_all.append(anatomy_flags)
            self.bags.append(images)
            self.texts.append(report)
            self.normal_flags_all.append(normal_flags)
            self.folder_names.append(folder)
            self.image_nums += len(images)

        if is_train:
            self.transform = self.get_transform(transform_list_file)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((testsize, testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # for i in range(len(self.abnormal_flags_all)):
        #     flag = self.abnormal_flags_all[i]
        #     sub_reports = self.texts[i].split('\n')[1:]
        #     if len(flag) != len(sub_reports):
        #         print(f"Error: flag length {len(flag)} != sub_reports length {len(sub_reports)}")
        #         print(sub_reports,'\n',flag)
        print('{} data: cases@ [{}], images@ [{}]'.format('train' if is_train else 'test', len(self.bags), self.image_nums))
        
    @staticmethod
    def get_transform(transform_list_file):
        transform_list = load_config(transform_list_file)
        tfs = []
        for key, value in transform_list.items():
            tf = eval(key)(**value) if value is not None else eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index):

        img_paths = self.bags[index]
        report_text = self.texts[index]
        folder_name = self.folder_names[index]
        normal_flags = self.normal_flags_all[index]
        anatomy_flags = self.anatomy_flags_all[index]

        # --- 策略：训练时随机采样，测试时全量或固定数量 ---
        if self.is_train and len(img_paths) > self.max_imgs:
            img_paths = random.sample(img_paths, self.max_imgs)
        
        images_tensor = []
        image_names = []
        for img_path in img_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                if self.is_train:
                    sample = self.transform({"image": image})
                    images_tensor.append(sample["image"])
                else:
                    images_tensor.append(self.transform(image))
                image_names.append(os.path.basename(img_path))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}. But don't worry because we will ignore this image and the bag has enough images.")
                continue

        lines = [line.strip() for line in report_text.split('\n') if line.strip()]
        sub_sentences = lines[1:] if len(lines) > 1 else lines 
        
        if self.mode in ['clip', 'mae']:
            full_tokens = self.tokenizer(report_text) 
            sub_tokens = self.tokenizer(sub_sentences) 
        else:
            full_tokens = self.tokenizer(report_text, context_length=256) 
            sub_tokens = self.tokenizer(sub_sentences, context_length=256) 

        return {
            "images": torch.stack(images_tensor), # [N_i, 3, H, W]
            "full_tokens": full_tokens.squeeze(0), # [77]
            "sub_tokens": sub_tokens,             # [Num_Sub, 77]
            "folder_name": folder_name,
            "image_names": image_names,
            "normal_flags": torch.tensor(normal_flags), # [Num_Sub]
            "anatomy_flags": torch.tensor(anatomy_flags), # [Num_Sub]
        } 
    
    def __len__(self):
        return len(self.bags)

# --- 关键：Dataloader 的加载逻辑 (Collate Function) ---
def mu_img_collate_fn(batch):
    """
    处理变长图片序列的逻辑：
    1. 将所有图片的张量拼接成一个大 Batch [Total_Imgs, 3, H, W]
    2. 记录每个 sample 包含的图片数量 image_counts
    3. 堆叠文本 Token
    """
    imgs_list = [item["images"] for item in batch]
    full_tokens_list = [item["full_tokens"] for item in batch]
    sub_tokens_list = [item["sub_tokens"] for item in batch]
    normal_flags = [item["normal_flags"] for item in batch]
    anatomy_flags = [item["anatomy_flags"] for item in batch]

    image_counts = [img.shape[0] for img in imgs_list]
    sub_text_counts = [sub.shape[0] for sub in sub_tokens_list]

    all_imgs = torch.cat(imgs_list, dim=0) # [Total_Image, 77]
    all_sub_texts = torch.cat(sub_tokens_list, dim=0) # [Total_sub_text, 77]
    all_normal_flags = torch.cat(normal_flags, dim=0) # [Total_sub_text]
    all_anatomy_flags = torch.cat(anatomy_flags, dim=0) # [Total_sub_text]

    all_full_texts = torch.stack(full_tokens_list, dim=0) # [B, 77]

    folders = [item["folder_name"] for item in batch] # [B]
    names = [item["image_names"] for item in batch] # [B]

    return {
        "images": all_imgs,         
        "full_texts": all_full_texts,
        "sub_texts": all_sub_texts,   
        "image_counts": image_counts, 
        "sub_text_counts": sub_text_counts,
        "folders": folders,
        "image_names": names,
        "normal_flags": all_normal_flags,
        "anatomy_flags": all_anatomy_flags,
    }


def get_train_loader_distributed(args, misc, tokenizer):
    # 1. Initialize Dataset
    if args.mode == 'mae':
        dataset = EndoscopyDataset_mae(
            data_file=args.data_file_train, 
            transform_list_file=args.transform_list_file, 
            tokenizer=tokenizer,
            is_train=True, 
            max_imgs=args.max_imgs, # Max images per patient during training
            debug=args.debug,
            debug_datasize=args.debug_datasize,
            testsize=args.input_size,
            mode=args.mode
        )
    else:
        dataset = EndoscopyDataset(
            data_file=args.data_file_train, 
            transform_list_file=args.transform_list_file, 
            tokenizer=tokenizer,
            is_train=True, 
            max_imgs=args.max_imgs, # Max images per patient during training
            debug=args.debug,
            debug_datasize=args.debug_datasize,
            testsize=args.input_size,
            mode=args.mode
        )

    # 2. Setup Distributed Sampler
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler = torch.utils.data.DistributedSampler(
            dataset, 
            num_replicas=num_tasks, 
            rank=global_rank, 
            shuffle=True,
            drop_last=True,
        )
        print(f"Sampler (Rank {global_rank}) initialized.")
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    # 3. Setup DataLoader with custom collate_fn
    # Note: 'shuffle' must be False when using a Sampler
    if args.mode == 'mae':
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            persistent_workers=True
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=mu_img_collate_fn, # CRITICAL: handles the multi-image flattening
            persistent_workers=True
        )

    return data_loader, dataset, sampler


class EndoscopyDataset_mae(data.Dataset):
    def __init__(self, data_file, transform_list_file, tokenizer, is_train=True, testsize=224, max_imgs=1000, debug=False, debug_datasize=100, mode='clip'):
        data_list = read_path_text_from_json(data_file)
        self.is_train = is_train
        self.tokenizer = tokenizer 
        self.max_imgs = max_imgs 
        self.mode = mode

        if debug:
            data_list = random.sample(data_list, debug_datasize)

        self.imgs = []
        self.image_nums = 0
        self.bag_num = 0
        for i, content in enumerate(data_list):
            # if i % (len(data_list) // 10+1) == 0:
            #     print(f"{i}/{len(data_list)}")
            folder_path = content['image_path']
            parts = folder_path.rstrip('/').split('/')
            folder = parts[-2] + '/' + parts[-1]

            # images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            #          if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '报告' not in f]
            images = [os.path.join(folder_path, f) for f in content['valid_image_paths']]
            report = content['endo_report']
            abnormal_flags = content['site_abnormality_flags']
            normal_flags = [1-x for x in abnormal_flags]
            sub_reports_len = len(report.split('\n')[1:])
            # ok, msg = check_image_ok(images[0])
            # if not ok: 
            #     print(msg)
            #     continue
            if len(images) == 0: continue
            if len(normal_flags) != sub_reports_len:
                print(f"Error: flag length {len(normal_flags)} != sub_reports length {sub_reports_len}")
                print(report,'\n',abnormal_flags)
                continue

            self.image_nums += len(images)
            self.imgs.extend(images)
            self.bag_num += 1

        if is_train:
            self.transform = self.get_transform(transform_list_file)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((testsize, testsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        # for i in range(len(self.abnormal_flags_all)):
        #     flag = self.abnormal_flags_all[i]
        #     sub_reports = self.texts[i].split('\n')[1:]
        #     if len(flag) != len(sub_reports):
        #         print(f"Error: flag length {len(flag)} != sub_reports length {len(sub_reports)}")
        #         print(sub_reports,'\n',flag)
        print('{} data: cases@ [{}], images@ [{}]'.format('train' if is_train else 'test', self.bag_num, self.image_nums))
        
    @staticmethod
    def get_transform(transform_list_file):
        transform_list = load_config(transform_list_file)
        tfs = []
        for key, value in transform_list.items():
            tf = eval(key)(**value) if value is not None else eval(key)()
            tfs.append(tf)
        return transforms.Compose(tfs)

    def __getitem__(self, index, _retry=0):
        img_path = self.imgs[index]
        try:
            image = Image.open(img_path).convert("RGB")

            if self.is_train:
                sample = self.transform({"image": image})
                images_tensor = sample["image"]          # 这里假设已经是 [N_i,3,H,W] 或 list[tensor]
            else:
                images_tensor = self.transform(image)

            image_name = os.path.basename(img_path)

            # if isinstance(images_tensor, torch.Tensor):
            #     stacked = images_tensor if images_tensor.dim() == 4 else images_tensor.unsqueeze(0)
            # else:
            #     stacked = torch.stack(images_tensor)

            return {"images": images_tensor, "image_names": image_name}

        except Exception as e:
            if _retry < 10:  # 最多重试10次，避免死循环
                new_index = random.randrange(len(self.imgs))
                return self.__getitem__(new_index, _retry=_retry + 1)
            # 重试还失败：抛出，让上层知道数据有严重问题
            raise RuntimeError(f"Failed to load after retries, last: {img_path}, err: {e}") from e

    
    def __len__(self):
        return len(self.imgs)