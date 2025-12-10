"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import re
import six
import math
import lmdb
import torch
import random
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, ConcatDataset, Subset
from utils import normalize_text, normalize_urdu_label

# _accumulate helper function (for compatibility with newer PyTorch versions)
def _accumulate(iterable):
    """Cumulative sum of iterable. Replaces torch._utils._accumulate for compatibility."""
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total += element
        yield total

# Callable classes for augmentation transforms (defined at module level for pickling compatibility)
class SaltPepperNoise:
    """Callable class wrapper for salt_and_pepper_noise to enable pickling on Windows."""
    def __call__(self, img):
        from modules.augmentation import salt_and_pepper_noise
        return salt_and_pepper_noise(img)

class RandomBorderCrop:
    """Callable class wrapper for random_border_crop to enable pickling on Windows."""
    def __call__(self, img):
        from modules.augmentation import random_border_crop
        return random_border_crop(img)

class RandomResize:
    """Callable class wrapper for random_resize to enable pickling on Windows."""
    def __call__(self, img):
        from modules.augmentation import random_resize
        return random_resize(img)

# PHASE 2: Callable classes for new targeted augmentations
class DotJitter:
    """Callable class wrapper for dot_jitter to enable pickling on Windows."""
    def __init__(self, prob=0.15):
        self.prob = prob
    def __call__(self, img):
        from modules.augmentation import dot_jitter
        return dot_jitter(img, prob=self.prob)

class SyntheticDotNoise:
    """Callable class wrapper for synthetic_dot_noise to enable pickling on Windows."""
    def __init__(self, prob=0.1):
        self.prob = prob
    def __call__(self, img):
        from modules.augmentation import synthetic_dot_noise
        return synthetic_dot_noise(img, prob=self.prob)

class HorizontalElasticDistortion:
    """Callable class wrapper for horizontal_elastic_distortion to enable pickling on Windows."""
    def __init__(self, max_shift=2, prob=0.2):
        self.max_shift = max_shift
        self.prob = prob
    def __call__(self, img):
        from modules.augmentation import horizontal_elastic_distortion
        return horizontal_elastic_distortion(img, max_shift=self.max_shift, prob=self.prob)

class StrokeThicknessVariation:
    """Callable class wrapper for stroke_thickness_variation to enable pickling on Windows."""
    def __init__(self, prob=0.15):
        self.prob = prob
    def __call__(self, img):
        from modules.augmentation import stroke_thickness_variation
        return stroke_thickness_variation(img, prob=self.prob)

class NumberAugmentation:
    """Callable class wrapper for number_augmentation to enable pickling on Windows."""
    def __init__(self, prob=0.3):
        self.prob = prob
    def __call__(self, img):
        from modules.augmentation import number_augmentation
        return number_augmentation(img, prob=self.prob)

class Batch_Balanced_Dataset(object):

    def __init__(self, opt, rand_aug = False):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a',encoding="utf-8")
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d], rand_aug=rand_aug)
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/', rand_aug = False):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    # print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt, rand_aug=rand_aug)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                # print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):
    # Process-local cache for LMDB environments (one per worker process)
    _env_cache = {}
    _cache_lock = None
    
    def __init__(self, root, opt,rand_aug=False, transform=None):
        self.root = root
        self.opt = opt
        self.rand_aug = rand_aug
        self.transform = transform
        # Don't store env directly - it can't be pickled on Windows
        # Instead, we'll open it lazily in each worker process
        
        # Initialize cache lock if not already done (thread-safe initialization)
        if LmdbDataset._cache_lock is None:
            import threading
            LmdbDataset._cache_lock = threading.Lock()
        
        # Get or create environment for this process
        env = self._get_env()
        if not env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            self.filtered_index_list = []
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')
                
                # CRITICAL FIX: Normalize Urdu label BEFORE normalize_text
                # This performs comprehensive Urdu-specific normalization
                raw_label = label
                label = normalize_urdu_label(label)
                if len(self.filtered_index_list) < 5:  # Print first 5 samples for verification
                    print(f"[DEBUG] Label normalization: {raw_label} → {label}")
                
                # CRITICAL FIX: Normalize label BEFORE filtering
                # This ensures Arabic/Persian variants are converted to Urdu canonical forms
                # and whitespace is normalized before checking length and character validity
                label = normalize_text(label)
                
                # CRITICAL FIX: Skip empty labels after normalization
                if len(label) == 0 or label.strip() == '':
                    continue

                if len(label) > self.opt.batch_max_length:
                    # print(f'The length of the label is longer than max_length: length {len(label)}, {label} in dataset {self.root}')
                    continue

                # By default, images containing characters which are not in opt.character are filtered.
                # You can add [UNK] token to `opt.character` in utils.py instead of this filtering
                out_of_char = f'[^{self.opt.character}]'
                
                # CRITICAL FIX: Filter characters and check if label becomes empty
                filtered_label = re.sub(out_of_char, '', label)
                if len(filtered_label) == 0 or filtered_label.strip() == '':
                    # Label becomes empty after filtering - skip this sample
                    continue
                
                if re.search(out_of_char, label):
                    # This warning is less critical now since we handle empty labels above
                    if len(self.filtered_index_list) < 10:  # Only print for first 10
                        print(f"⚠️  Sample {index}: Contains out-of-vocab chars, but filtered label is valid")
                    # Continue - we'll use the filtered label in __getitem__

                self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)
        
        if self.transform is None:
            self.transform = []
        if self.rand_aug:
            from modules.augmentation import rand_augment_transform
            self.transform.append(rand_augment_transform())
            self.transform.append(T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25))
            if random.random()<0.25:
                # Use module-level callable class instead of lambda for pickling compatibility
                self.transform.append(SaltPepperNoise())
            if random.random()<0.25:
                # Use module-level callable class instead of lambda for pickling compatibility
                self.transform.append(RandomBorderCrop())
            self.transform.append(T.RandomRotation(2))  # Reduced from 5 to 2 degrees to preserve Urdu diacritics
            if random.random()<0.25:
                # Use module-level callable class instead of lambda for pickling compatibility
                self.transform.append(RandomResize())
            
            # PHASE 2: Add targeted augmentations for Urdu OCR error patterns
            # Dot Jitter: Helps with dot detection failures (prob=0.15)
            self.transform.append(DotJitter(prob=0.15))
            
            # Synthetic Dot Noise: Directly attacks dot confusion (prob=0.1)
            if random.random() < 0.1:
                self.transform.append(SyntheticDotNoise(prob=1.0))  # Apply if selected
            
            # Horizontal Elastic Distortion: Handles ligature breakage (prob=0.2)
            if random.random() < 0.2:
                self.transform.append(HorizontalElasticDistortion(max_shift=2, prob=1.0))
            
            # Stroke Thickness Variation: Font style variations (prob=0.15)
            if random.random() < 0.15:
                self.transform.append(StrokeThicknessVariation(prob=1.0))
            
            # Note: Number augmentation is applied conditionally in __getitem__ based on label
            self.transform = T.Compose(self.transform)
    
    def _get_env(self):
        """Get or create LMDB environment for this process (cached per process)."""
        # Initialize lock if not already done (needed for worker processes)
        if LmdbDataset._cache_lock is None:
            import threading
            LmdbDataset._cache_lock = threading.Lock()
        
        # Use process ID as part of cache key to ensure each worker process has its own env
        import os
        cache_key = (os.getpid(), self.root)
        
        if cache_key not in LmdbDataset._env_cache:
            with LmdbDataset._cache_lock:
                # Double-check after acquiring lock
                if cache_key not in LmdbDataset._env_cache:
                    env = lmdb.open(self.root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
                    LmdbDataset._env_cache[cache_key] = env
        return LmdbDataset._env_cache[cache_key]

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        env = self._get_env()
        with env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            
            # CRITICAL FIX: Normalize Urdu label BEFORE normalize_text
            # This performs comprehensive Urdu-specific normalization
            label = normalize_urdu_label(label)
            
            # CRITICAL FIX: Normalize label BEFORE filtering
            # This ensures Arabic/Persian variants are converted to Urdu canonical forms
            # and whitespace is normalized before character filtering
            label = normalize_text(label)
            
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img1 = Image.open(buf).convert('RGB')
                    img = img1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # Flip for UTRNet-Large compatibility
                else:
                    img1 = Image.open(buf).convert('L')
                    img = img1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # Flip for UTRNet-Large compatibility

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img1 = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                    img = img1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # Flip for UTRNet-Large compatibility
                else:
                    img1 = Image.new('L', (self.opt.imgW, self.opt.imgH))
                    img = img1.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # Flip for UTRNet-Large compatibility
                label = '[dummy_label]'

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            # Note: Label is already normalized above, so this filtering happens on normalized text
            out_of_char = f'[^{self.opt.character}]'
            removed_chars = set(re.findall(out_of_char, label))
            if removed_chars and index <= 10:  # Warn for first 10 samples
                print(f"⚠️  Removed chars from label at index {index}: {removed_chars}")
            label = re.sub(out_of_char, '', label)
            
            # CRITICAL FIX: Skip empty labels after filtering
            # Empty labels cause CTC loss issues and prevent learning
            if len(label) == 0 or label.strip() == '':
                # Return a dummy sample with a valid label to prevent crashes
                # This should be rare since we filter in __init__, but handle it gracefully
                if index <= 10:
                    print(f"⚠️  WARNING: Empty label after filtering at index {index}, using fallback")
                # Use a common Urdu word as fallback (should never happen if __init__ filtering works)
                label = 'ا'  # Single character 'alif' as minimal valid label

            if self.transform:
                img = self.transform(img)
            
            # PHASE 2: Apply number-specific augmentation if label contains numbers
            # Check if label contains any digits (0-9)
            if self.rand_aug and any(char.isdigit() for char in label):
                if random.random() < 0.3:  # 30% probability for numeric samples
                    img = NumberAugmentation(prob=1.0)(img)
            
            # PHASE 3: Apply text-level augmentation for spacing and number issues
            # This addresses common errors: spacing variations, number recognition, punctuation spacing
            if self.rand_aug:
                from modules.augmentation import apply_text_augmentation
                # Store original label as backup
                original_label = label
                # Apply text augmentation with moderate probabilities
                # Only apply if label is not too short (avoid breaking very short labels)
                if len(label) > 3:
                    label = apply_text_augmentation(
                        label,
                        spacing_prob=0.3,          # 30% chance for spacing variations
                        number_spacing_prob=0.7,    # 70% chance for number spacing (if numbers present)
                        punctuation_prob=0.4,        # 40% chance for punctuation spacing
                        number_format_prob=0.6       # 60% chance for number format variations
                    )
                    # Re-filter after augmentation to ensure all characters are still valid
                    label = re.sub(out_of_char, '', label)
                    # Ensure label is not empty after augmentation
                    if len(label) == 0 or label.strip() == '':
                        # If augmentation broke the label, use original
                        label = original_label

        return (img, label)

class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = T.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = T.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=48, imgW=100, keep_ratio_with_pad=True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
