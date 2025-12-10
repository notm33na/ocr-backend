"""
Paper: "UTRNet: High-Resolution Urdu Text Recognition In Printed Documents" presented at ICDAR 2023
Authors: Abdur Rahman, Arjun Ghosh, Chetan Arora
GitHub Repository: https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
Project Website: https://abdur75648.github.io/UTRNet/
Copyright (c) 2023-present: This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License (http://creativecommons.org/licenses/by-nc/4.0/)
"""

import pytz
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from torch.autograd import Variable

import os,random,shutil
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Try to import editdistance library, fallback to nltk
try:
    import editdistance
    def levenshtein_distance(s1, s2):
        """Compute Levenshtein distance between two strings using editdistance library."""
        return editdistance.eval(s1, s2)
except ImportError:
    # Fallback to nltk if editdistance is not available
    try:
        from nltk.metrics.distance import edit_distance
        def levenshtein_distance(s1, s2):
            """Compute Levenshtein distance between two strings using nltk."""
            return edit_distance(s1, s2)
    except ImportError:
        # Final fallback: simple implementation
        def levenshtein_distance(s1, s2):
            """Simple Levenshtein distance implementation."""
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            return previous_row[-1]

def word_levenshtein_distance(words1, words2):
    """Compute Levenshtein distance between two word lists (for WER calculation)."""
    # Handle empty cases
    if len(words1) == 0:
        return len(words2)
    if len(words2) == 0:
        return len(words1)
    
    # Use dynamic programming for word-level Levenshtein
    m, n = len(words1), len(words2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i-1] == words2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + 1     # substitution
                )
    
    return dp[m][n]


def normalize_urdu_label(text):
    """
    Normalize Urdu labels with comprehensive Urdu-specific normalization rules.
    This function should be called BEFORE length filtering, charset filtering, and training dataset creation.
    
    Steps:
    1. Convert Arabic variants to Urdu canonical forms
    2. Normalize diacritics (remove tashkeel)
    3. Normalize punctuation
    4. Remove invisible Unicode characters
    5. Strip extra whitespace
    
    Args:
        text (str): Input Urdu text label
        
    Returns:
        str: Normalized Urdu text
    """
    import unicodedata
    import re
    
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: Convert Arabic variants to Urdu canonical forms
    # Replace 'ي' → 'ی'
    text = text.replace('ي', 'ی')
    # Replace 'ك' → 'ک'
    text = text.replace('ك', 'ک')
    # Replace 'ة' or 'ۃ' → 'ہ'
    text = text.replace('ة', 'ہ')
    text = text.replace('ۃ', 'ہ')
    # Replace 'ئ' forms consistently: 'ئى', 'یٔ', 'ئ' → 'ئ'
    text = re.sub(r'ئى|یٔ', 'ئ', text)
    # Keep 'ئ' as is (already correct)
    
    # Step 2: Normalize diacritics - Remove tashkeel
    # Remove: َ ً ُ ِ ٓ ٔ ٖ ٘ ّ  ٰ ٗ ْ ٌ ٍ
    tashkeel_chars = ['َ', 'ً', 'ُ', 'ِ', 'ٓ', 'ٔ', 'ٖ', '٘', 'ّ', 'ٰ', 'ٗ', 'ْ', 'ٌ', 'ٍ']
    for char in tashkeel_chars:
        text = text.replace(char, '')
    
    # Step 3: Normalize punctuation
    # Replace Arabic comma '،' → ',' (keep as is for Urdu, but normalize spacing)
    # Replace Arabic question '؟' → '?' (keep as is for Urdu, but normalize spacing)
    # Replace full stop '۔' → '۔' (keep single canonical form - already correct)
    # Note: We keep Urdu punctuation marks but normalize spacing around them
    
    # Step 4: Remove invisible Unicode characters
    # Strip zero-width joiners (U+200D)
    text = text.replace('\u200D', '')
    # Strip zero-width non-joiners (U+200C)
    text = text.replace('\u200C', '')
    # Strip RTL markers (U+200F, U+202B, U+202E)
    text = text.replace('\u200F', '')
    text = text.replace('\u202B', '')
    text = text.replace('\u202E', '')
    # Strip left-to-right mark (U+200E)
    text = text.replace('\u200E', '')
    
    # Step 5: Strip extra whitespace and collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces to single space
    text = text.strip()  # Strip leading/trailing whitespace
    
    return text


def normalize_text(s):
    """
    Normalize text for consistent evaluation with Urdu-specific rules.
    
    Steps:
    1. Unicode NFKC normalization (decomposes and recomposes characters)
    2. Character equivalence mapping (Unicode variants to canonical forms)
    3. Collapse repeated whitespace to single space
    4. Normalize spacing around punctuation (Urdu-specific)
    5. Strip leading and trailing whitespace
    
    Args:
        s (str): Input text string
        
    Returns:
        str: Normalized text
    """
    import unicodedata
    import re
    
    if not isinstance(s, str):
        s = str(s)
    
    # Step 1: Unicode NFKC normalization
    s = unicodedata.normalize('NFKC', s)
    
    # Step 2: Character equivalence mapping (Urdu-specific Unicode variants)
    # Map Arabic/Persian variants to Urdu canonical forms
    char_equivalences = {
        'ي': 'ی',  # Arabic yeh (U+064A) → Urdu yeh (U+06CC)
        'ى': 'ی',  # Arabic alef maksura (U+0649) → Urdu yeh
        'ك': 'ک',  # Arabic kaf (U+0643) → Urdu kaf (U+06A9)
        'ة': 'ہ',  # Arabic teh marbuta (U+0629) → Urdu heh (U+06C1)
        'أ': 'ا',  # Arabic alef with hamza above → Urdu alef
        'ؤ': 'و',  # Arabic waw with hamza → Urdu waw
        'إ': 'ا',  # Arabic alef with hamza below → Urdu alef
        'ئ': 'ی',  # Arabic yeh with hamza → Urdu yeh
    }
    for old_char, new_char in char_equivalences.items():
        s = s.replace(old_char, new_char)
    
    # Step 3: Normalize spacing around punctuation (Urdu-specific)
    # Remove space before punctuation marks
    s = re.sub(r'\s+([۔،؛:؟!])', r'\1', s)  # Space before punctuation → no space
    # Ensure space after punctuation (if not at end)
    s = re.sub(r'([۔،؛:؟!])([^\s])', r'\1 \2', s)  # Punctuation followed by non-space → add space
    # Remove space before closing brackets/parentheses
    s = re.sub(r'\s+([\)\]\}])', r'\1', s)
    # Ensure space after opening brackets/parentheses
    s = re.sub(r'([\(\[\{])([^\s])', r'\1 \2', s)
    
    # Step 4: Collapse all whitespace variations to single space
    # This handles: multiple spaces, tabs, newlines, etc.
    s = ' '.join(s.split())
    
    # Step 5: RTL-specific spacing normalization
    # Fix common spacing issues in compound words (but preserve word boundaries)
    # Don't over-normalize - let the model learn some variations
    
    # Step 6: Strip leading and trailing spaces
    s = s.strip()
    
    return s


def post_process_ocr_output(text):
    """
    Post-process OCR output to normalize spacing and punctuation.
    
    This function addresses common OCR errors observed in predictions:
    - Spacing issues: "کار روایی" → "کارروایی", "کویقینی" → "کو یقینی"
    - Punctuation spacing: ")اے ایف پی(" → ") اے ایف پی ("
    - Number spacing: "1000 سے" → "1000 سے" (normalize)
    - Common compound word spacing errors
    
    Args:
        text (str): Raw OCR prediction output
        
    Returns:
        str: Post-processed text with normalized spacing and punctuation
    """
    import re
    
    if not isinstance(text, str) or len(text) == 0:
        return text
    
    # Step 1: Normalize basic text first (Unicode, character variants)
    text = normalize_text(text)
    
    # Step 2: Fix spacing in common compound words (based on training log errors)
    # These patterns fix spacing issues where model incorrectly adds/removes spaces
    # Be conservative - only fix known error patterns
    
    # Fix: "کار روایی" → "کارروایی" (remove incorrect space in compound word)
    text = re.sub(r'\bکار\s+روایی\b', 'کارروایی', text)
    
    # Fix: "کویقینی" → "کو یقینی" (add missing space)
    text = re.sub(r'\bکو(یقینی)\b', r'کو \1', text)
    
    # Fix: "تھاکہ" → "تھا کہ" (add space before کہ when it's a separate word)
    # Only fix if preceded by a word (not at start) and کہ is followed by space or end
    text = re.sub(r'(\w+)(کہ)(\s|$)', r'\1 \2\3', text)
    
    # Fix: "زپر" → "ز پر" (add space - this was a common error)
    # Only if "ز" is a single character followed by "پر"
    text = re.sub(r'\b(\w)(پر)(\s|$)', r'\1 \2\3', text)
    
    # Note: We don't fix "اور اس" because "اور اس" is usually correct
    # (meaning "and this/that")
    
    # Step 3: Normalize punctuation spacing
    # Fix: ")اے ایف پی(" → ") اے ایف پی ("
    # Add space after closing brackets/parentheses if followed by letter
    text = re.sub(r'([\)\]\}])([^\s\)\]\}\(\[\{\.\,\:\;])', r'\1 \2', text)
    
    # Add space before opening brackets/parentheses if preceded by letter
    text = re.sub(r'([^\s\)\]\}\(\[\{\.\,\:\;])([\(\[\{])', r'\1 \2', text)
    
    # Remove space before closing brackets (should be attached)
    text = re.sub(r'\s+([\)\]\}])', r'\1', text)
    
    # Ensure space after opening brackets (if not at start)
    text = re.sub(r'([\(\[\{])([^\s\)\]\}\(\[\{])', r'\1 \2', text)
    
    # Step 4: Normalize spacing around numbers
    # Ensure space after numbers when followed by Urdu text
    text = re.sub(r'(\d+)([^\d\s\.\,\:\;])', r'\1 \2', text)
    
    # Ensure space before numbers when preceded by Urdu text
    text = re.sub(r'([^\d\s\.\,\:\;])(\d+)', r'\1 \2', text)
    
    # But don't add space between number and common punctuation
    text = re.sub(r'(\d+)\s+([\.\,\:\;])', r'\1\2', text)
    
    # Step 5: Normalize spacing around Urdu punctuation marks
    # Urdu punctuation: ۔ (full stop), ، (comma), ؛ (semicolon), : (colon), ؟ (question mark)
    # Remove space before punctuation
    text = re.sub(r'\s+([۔،؛:؟!])', r'\1', text)
    
    # Ensure space after punctuation (if not at end of string)
    text = re.sub(r'([۔،؛:؟!])([^\s۔،؛:؟!])', r'\1 \2', text)
    
    # Step 6: Fix common spacing errors in compound words
    # These are based on observed errors in training log
    
    # Fix: "ایمپلار زپر" → "ایمپلایرز پر" (this is a character error, not spacing)
    # Actually, this is a recognition error, not spacing - leave it
    
    # Fix: "کویقینی" → "کو یقینی" (already handled above)
    
    # Step 7: Collapse multiple spaces to single space
    text = re.sub(r' +', ' ', text)
    
    # Step 8: Remove spaces at start/end of string
    text = text.strip()
    
    # Step 9: Fix common OCR artifacts
    # Remove isolated single characters that are likely OCR errors
    # (But be conservative - only remove if it's clearly an artifact)
    
    return text


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts


class CTCLabelConverterForBaiduWarpctc(object):
    """ Convert between text-label and text-index for baidu warpctc """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
            index += l
        return texts


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length=25):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.LongTensor(len(text), batch_max_length + 1).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            
            try:
                text = [self.dict[char] for char in text]
            except KeyError as e:
                continue
            batch_text[i][1:1 + len(text)] = torch.LongTensor(text)  # batch_text[:, 0] = [GO] token
        return (batch_text, torch.IntTensor(length))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            # CRITICAL FIX: Only decode up to length l, and strip [GO] and [s] tokens
            text_chars = []
            for i in range(min(l, text_index.size(1))):
                char_idx = text_index[index, i].item()
                if char_idx < len(self.character):
                    char = self.character[char_idx]
                    # Skip [GO] token (index 0)
                    if char != '[GO]':
                        # Stop at [s] token (end of sentence)
                        if char == '[s]':
                            break
                        text_chars.append(char)
            text = ''.join(text_chars)
            texts.append(text)
        return texts


def imshow(img, title,batch_size=1):
  std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)
  mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)
  npimg = np.multiply(img.numpy(), std_correction) + mean_correction
  plt.figure(figsize = (batch_size * 4, 4))
  plt.axis("off")
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.title(title)
  plt.show()


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""

    def __init__(self):
        self.reset()

    def add(self, v):
        count = v.data.numel()
        v = v.data.sum()
        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

class Logger(object):
    """For logging while training"""
    def __init__(self, path):
        self.logFile = path
        datetime_now = str(datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%Y-%m-%d_%H-%M-%S"))
        with open(self.logFile,"w",encoding="utf-8") as f:
            f.write("Logging at @ " + str(datetime_now) + "\n")

    def log(self,*input):
        message = ""
        for x in input:
            message+=str(x) + " "
        message = message.strip()
        print(message)
        with open(self.logFile,"a",encoding="utf-8") as f:
            f.write(str(message)+"\n")


def allign_two_strings(x:str, y:str, pxy:int=1, pgap:int=1):
    """
    Source: https://www.geeksforgeeks.org/sequence-alignment-problem/
    """
    i = 0
    j = 0
    m = len(x)
    n = len(y)
    dp = np.zeros([m+1,n+1], dtype=int)
    dp[0:(m+1),0] = [ i * pgap for i in range(m+1)]
    dp[0,0:(n+1)] = [ i * pgap for i in range(n+1)]
 
    i = 1
    while i <= m:
        j = 1
        while j <= n:
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1] + pxy,
                                dp[i - 1][j] + pgap,
                                dp[i][j - 1] + pgap)
            j += 1
        i += 1
     
    l = n + m 
    i = m
    j = n
     
    xpos = l
    ypos = l
 
    xans = np.zeros(l+1, dtype=int)
    yans = np.zeros(l+1, dtype=int)
 
    while not (i == 0 or j == 0):
        #print(f"i: {i}, j: {j}")
        if x[i - 1] == y[j - 1]:       
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
        elif (dp[i - 1][j - 1] + pxy) == dp[i][j]:
         
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            i -= 1
            j -= 1
         
        elif (dp[i - 1][j] + pgap) == dp[i][j]:
            xans[xpos] = ord(x[i - 1])
            yans[ypos] = ord('_')
            xpos -= 1
            ypos -= 1
            i -= 1
         
        elif (dp[i][j - 1] + pgap) == dp[i][j]:       
            xans[xpos] = ord('_')
            yans[ypos] = ord(y[j - 1])
            xpos -= 1
            ypos -= 1
            j -= 1
         
 
    while xpos > 0:
        if i > 0:
            i -= 1
            xans[xpos] = ord(x[i])
            xpos -= 1
        else:
            xans[xpos] = ord('_')
            xpos -= 1
     
    while ypos > 0:
        if j > 0:
            j -= 1
            yans[ypos] = ord(y[j])
            ypos -= 1
        else:
            yans[ypos] = ord('_')
            ypos -= 1

    id = 1
    i = l
    while i >= 1:
        if (chr(yans[i]) == '_') and chr(xans[i]) == '_':
            id = i + 1
            break
         
        i -= 1
 
    i = id
    x_seq = ""
    while i <= l:
        x_seq += chr(xans[i])
        i += 1
 
    # Y
    i = id
    y_seq = ""
    while i <= l:
        y_seq += chr(yans[i])
        i += 1
    
    return x_seq, y_seq

# Function to count the number of trainable parameters in a model in "Millions"
def count_parameters(model,precision=2):
    return (round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 10.**6, precision))

'''
# Code for counting the number of FLOPs in the CNN backbone during inference
Source - https://github.com/fdbtrs/ElasticFace/blob/main/utils/countFLOPS.py
'''

def count_model_flops(model,in_channels=1, input_res=[32, 400], multiply_adds=True):
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (
            2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size
        list_conv.append(flops)
    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        if self.bias is not None:
            bias_ops = self.bias.nelement() if self.bias.nelement() else 0
            flops = batch_size * (weight_ops + bias_ops)
        else:
            flops = batch_size * weight_ops
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        # If kernel_size is a tuple type, computer ops as product of elements or else if it is int type, compute ops as square of kernel_size
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] if isinstance(self.kernel_size, tuple) else self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size
        list_pooling.append(flops)
    
    def dropout_hook(self, input, output):
        # calculate the number of operations for a dropout function by assuming that each operation involves one comparison and one multiplication
        batch_size, input_channels, input_height, input_width = input[0].size()
        list_conv.append(2*batch_size*input_channels*input_height*input_width)
    
    def sigmoid_hook(self,input,output):
        # calculate the number of operations for a sigmoid function by assuming that each operation involves two multiplications and one addition
        batch_size, input_channels, input_height, input_width = input[0].size()
        list_conv.append(3*batch_size*input_channels*input_height*input_width)
    
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.scale_factor * self.scale_factor # * (self.in_channels / self.groups)
        flops = (kernel_ops * (
            2 if multiply_adds else 1)) * output_channels * output_height * output_width * batch_size
        list_conv.append(flops)

    handles = []

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                handles.append(net.register_forward_hook(conv_hook))
            elif isinstance(net, torch.nn.Linear):
                handles.append(net.register_forward_hook(linear_hook))
            elif isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.BatchNorm1d):
                handles.append(net.register_forward_hook(bn_hook))
            elif isinstance(net, torch.nn.ReLU) or isinstance(net, torch.nn.PReLU):
                handles.append(net.register_forward_hook(relu_hook))
            elif isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                handles.append(net.register_forward_hook(pooling_hook))
            elif isinstance(net, torch.nn.Dropout):
                handles.append(net.register_forward_hook(dropout_hook))
            elif isinstance(net,torch.nn.Upsample):
                handles.append(net.register_forward_hook(upsample_hook))
            elif isinstance(net,torch.nn.Sigmoid):
                handles.append(net.register_forward_hook(sigmoid_hook))
            else:
                print("warning" + str(net))
            return
        for c in childrens:
            foo(c)

    model.eval()
    foo(model)
    input = Variable(torch.rand(in_channels, input_res[1], input_res[0]).unsqueeze(0), requires_grad=True)
    out = model(input)
    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    for h in handles:
        h.remove()
    model.train()
    
    def flops_to_string(flops, units='MFLOPS', precision=4):
        if units == 'GFLOPS':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MFLOPS':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KFLOPS':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' FLOPS'
    
    return flops_to_string(total_flops)


def draw_feature_map(visual_feature,vis_dir,num_channel=10):
    """draws feature maps for the given visual features
    Args:
        visual_feature (Tensor): Shape (C, H, W)
        vis_dir (String): Directory to save the feature maps
    """
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)
    # Save visual_feature from num_channel random channels for visualization
    for i in range(num_channel):
        random_channel = random.randint(0, visual_feature.shape[1]-1)
        visual_feature_for_visualization = visual_feature[0, random_channel, :, :].detach().cpu().numpy()
        # Horizontal flip
        visual_feature_for_visualization = visual_feature_for_visualization[:,::-1]
        # Normalize
        visual_feature_for_visualization = (visual_feature_for_visualization - visual_feature_for_visualization.min()) / (visual_feature_for_visualization.max() - visual_feature_for_visualization.min())
        # Draw heatmap
        plt.imshow(visual_feature_for_visualization, cmap='gray', interpolation='nearest')
        plt.axis("off")
        plt.savefig(os.path.join(vis_dir, "channel_{}.png".format(random_channel)), bbox_inches='tight', pad_inches=0)